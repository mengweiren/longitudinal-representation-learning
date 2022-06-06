from ast import Or, Pass
from distutils.log import debug
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.base_model import BaseModel
from . import networks
from models.networks import Pairwise_Feature_Similarity
import util.util as util
from collections import OrderedDict
import os
import monai

class LongRepModel(BaseModel):
    """
    The code borrows heavily from the PyTorch implementation of CUT
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for PatchSim + Seg model
        """
        parser.add_argument('--lambda_seg', type=float, default=1.0, help='weight for segmentation loss: L1(G(x) S)')
        parser.add_argument('--lambda_cseg', type=float, default=1.0, help='weight for segmentation consistency')    
        parser.add_argument('--lambda_sim', type=float, default=1.0, help='weight for Sim loss: Sim(X,Y)')

        parser.add_argument('--unfreeze_layers', type=str, default='', help='specify partial layers that will be trained')
    
        parser.add_argument('--netF', type=str, default='simsiam_mlp_sample', choices=['simsiam_mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--archF', type=int, default=2, choices=[1,2], help='delete later, 1 is the previous model with bias/BN affine, 2 is the same as simsiam github implementation ')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--num_patches', type=int, default=768, help='number of patches per layer')
        parser.add_argument('--n_mlps', type=int, default=3, help='number of mlps, 2 or 3')

        parser.add_argument('--sim_layers', type=str, default='3,10,17,24,27,31,38,45,52',
                            help='compute Sim loss on which layers')
        parser.add_argument('--sim_weights', type=str, default='1',
                            help='how to sum all nce losses. If "1", simply use mean. Otherwise, weighted average by specified weights that must have equal length of sim_layers')
        parser.add_argument('--use_mlp', type=util.str2bool, nargs='?', const=True, default=True, help='whether to use MLP ')
        
        parser.add_argument('--reg_V', type=str, default='0', help='regularization weight for batch variance')
        parser.add_argument('--reg_C', type=str, default='0', help='regularization weight for dimension covariance')
        parser.add_argument('--reg_O', type=str, default='0', help='regularization weight for orthogonal encoder decoder')

        parser.add_argument('--ortho_idx', type=str, default='0,1,2',
                    help='compute orthogonal loss on which concat pairs')

        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G']  # total loss 
        if self.opt.lambda_seg > 0:
            self.loss_names += ['S'] #, 'dice', 'ce']
            assert self.opt.lambda_Rec == 0., 'segmentation and reconstruction conflicts.'
            self.visual_names = ['real_A', 'pred_seg', 'seg_A']

        if self.opt.lambda_Rec > 0:
            self.loss_names += ['Rec']
            assert self.opt.lambda_seg == 0., 'segmentation and reconstruction conflicts.'
            self.visual_names = ['real_A', 'rec_A','rec_A_trg']


        if self.isTrain:
            self.grad_accum_iters=opt.grad_accum_iters

            if self.opt.lambda_cseg > 0.:
                self.loss_names += ['CS']
                self.visual_names = ['real_A', 'pred_seg_A', 'real_B', 'pred_seg_B']

            if self.opt.lambda_sim > 0:
                self.loss_names += ['Sim','std','cov','ortho']

        print('Visuals: {}'.format(self.visual_names))

        if not opt.sim_layers == '':
            self.sim_layers = [int(i) for i in self.opt.sim_layers.split(',')]
            self.last_layer_to_freeze = self.sim_layers[-1]
        else:
            self.sim_layers = []
            self.last_layer_to_freeze = -1

        if self.opt.sim_weights != '1':
            self.sim_weights = [float(i) for i in self.opt.sim_weights.split(',')]
            sum_ = np.sum(np.asarray(self.sim_weights))
            self.sim_weights = [i / sum_ for i in self.sim_weights]
        else:
            if len(self.sim_layers) > 0:
                self.sim_weights = [1./len(self.sim_layers) for _ in self.sim_layers]
            else:
                self.sim_weights = []
        assert len(self.sim_weights) == len(self.sim_layers)

        self.reg_V =  [float(i) for i in self.opt.reg_V.split(',')]
        self.reg_C =  [float(i) for i in self.opt.reg_C.split(',')]
        self.reg_O =  float(self.opt.reg_O)

        if len(self.reg_V) == 1:
            self.reg_V = [self.reg_V[0]  for _ in self.sim_layers]
        if len(self.reg_C) == 1:
            self.reg_C = [self.reg_C[0]  for _ in self.sim_layers]
        #if len(self.reg_O) == 1:
            #self.reg_O = [self.reg_O[0]  for _ in self.sim_layers]
        for i in range(len(self.sim_layers)):
            self.reg_V[i] = self.reg_V[i] * self.sim_weights[i]  # scale by the weight 
            self.reg_C[i] = self.reg_C[i] * self.sim_weights[i]
            #self.reg_O[i] = self.reg_O[i] * self.sim_weights[i]


        if self.opt.lambda_sim > 0:
            print('--------------- Loss configuration -------------')
            print('sim Layers', self.sim_layers)
            print('sim weights', self.sim_weights)
            print('regularization (V) - std', self.reg_V)
            print('regularization (C) - cov', self.reg_C)

        self.reset_params = False
        if not self.opt.unfreeze_layers == '':
            self.unfreeze_layers = [i for i in self.opt.unfreeze_layers.split(',')]
            self.reset_params = True
            #assert self.opt.pretrained_name is not None, 'Must have pretrained checkpoints for the freezed layers!!'
            self.load_model_names = ['G']
        self.model_names = ['G']
        # define networks (generator)
        self.netG = networks.define_G(opt.ndims, opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, self.gpu_ids, opt, pooling=opt.pool_type, interp=opt.interp_type,
                                      final_act='none', activation=opt.actG)
        
        ### Get the paired encoder and decoder features for orthogonal regularization 
        enc_idx = self.netG.encoder_idx
        dec_idx = self.netG.decoder_idx[::-1]
        #Encoder skip connect id [8, 15, 22, 29]
        #Decoder skip connect id [37, 44, 51, 58]
        all_pairs = []
        for eid, did in zip(enc_idx, dec_idx):
            #if (eid) in self.sim_layers and (did+3) in self.sim_layers:  # after relu 
            if (eid-2) in self.sim_layers and (did+1) in self.sim_layers: # after conv
                print(eid, did)
                nce_e = self.sim_layers.index(eid-2)
                nce_d = self.sim_layers.index(did+1)
                all_pairs.append((nce_e,nce_d))  # the encoder/decoder id in sim_layer 
        
        print('Found all concat pairs', all_pairs)
        self.ortho_idx = [int(i) for i in self.opt.ortho_idx.split(',')]
        if -1 in self.ortho_idx:
            self.ortho_idx.append(len(all_pairs)-1)
        print('using {} as orthogonal loss'.format(self.ortho_idx))
        self.concat_pairs = []
        for i in range(len(all_pairs)):
            if i in self.ortho_idx:
                self.concat_pairs.append(all_pairs[i])

        self.reg_O /= max(len(self.concat_pairs),1)
        print('regularization (O) - Ortho', self.reg_O)
        print('apply orthogonal loss on {}'.format(self.concat_pairs))
        for e,d in self.concat_pairs:
            print('ID', self.sim_layers[e], self.sim_layers[d])

        if not os.path.exists(os.path.join(self.save_dir, 'netG.txt')):
            print('Write network configs into text')
            with open(os.path.join(self.save_dir, 'netG.txt'), 'w') as f:
                print(self.netG, file=f)
            f.close()

        if self.opt.lambda_Rec > 0:
            self.criterionRec = torch.nn.MSELoss()
            self.model_names += ['Rec']
            self.netRec = networks.define_G(opt.ndims, opt.output_nc, opt.input_nc, opt.ngf, 'conv', opt.normG,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt,
                                            final_act='none')
            if not os.path.exists(os.path.join(self.save_dir, 'netRec.txt')):
                print('Write recon net configs into text')
                with open(os.path.join(self.save_dir, 'netRec.txt'), 'w') as f:
                    print(self.netRec, file=f)
                f.close()

        if self.isTrain:
            if self.opt.lambda_sim > 0.:
                self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                              opt.init_gain, opt.n_mlps, self.gpu_ids, opt, activation=opt.actF, arch=opt.archF, use_mlp=opt.use_mlp)

                if self.opt.use_mlp:
                    self.model_names += ['F']

                # define loss functions
                self.criterionSim = []
                for sim_layer in self.sim_layers:
                    self.criterionSim.append(Pairwise_Feature_Similarity(opt).to(self.device))
 
            lambda_dice = 1.
            lambda_ce = 1
            smooth_dr = self.opt.smooth_dr
            smooth_nr = self.opt.smooth_nr
            batch_dice= self.opt.batch_dice
            include_background=self.opt.include_background

            print(f'Dice {lambda_dice}, CE {lambda_ce}, smooth_dr {smooth_dr}, smooth_nr {smooth_nr}, batch {batch_dice}') #, include_background {include_background}')

            self.criterionSegC = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
            
            self.criterionSeg = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=lambda_dice, lambda_ce=lambda_ce,
                                                        smooth_nr=smooth_nr, smooth_dr=smooth_dr, batch=batch_dice)
                                                        
            self.criterionSeg_val = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=lambda_dice, lambda_ce=lambda_ce, smooth_nr=1e-5, smooth_dr=1e-5)

            if self.reset_params:
                paramsG = []
                params_dict_G = dict(self.netG.named_parameters())
                for key, value in params_dict_G.items():
                    grad = False
                    for f in self.unfreeze_layers:
                        if f in key:
                            print('Add %s to optimizer list'%key)
                            grad = True
                            paramsG += [{'params': [value]}]
                            break
                    value.requires_grad = grad
                    print(key, value.requires_grad)

            else:
                paramsG = self.netG.parameters()

            if self.opt.lambda_Rec > 0:
                print('Adding netRec parameters for reconstruction')
                import itertools
                paramsG = itertools.chain(paramsG, self.netRec.parameters())
            
            #self.optimizer_G =  torch.optim.AdamW(paramsG, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay,
            #                                      amsgrad=opt.ams_grad)
            self.optimizer_G = torch.optim.Adam(paramsG, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            
            self.optimizers.append(self.optimizer_G)

        else:
            self.feat_names = ['feat','global_feat']

        self.new_bs = -1
        if len(self.load_model_names) == 0:
            self.load_model_names = self.model_names


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        if self.opt.isTrain:
            if self.opt.lambda_sim > 0:
                self.set_input(data, verbose=True)
                self.forward(verbose=False)                     # compute fake images: G(A)
                self.compute_G_loss(0).backward()                   # calculate graidents for G
                
                if self.opt.lambda_sim > 0.0 and self.opt.use_mlp and not self.freeze_netF:
                    self.optimizer_F = torch.optim.AdamW(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2),
                                                         eps=self.opt.eps, weight_decay=self.opt.weight_decay,amsgrad=self.opt.ams_grad)
                    self.optimizers.append(self.optimizer_F)
                if not os.path.exists(os.path.join(self.save_dir, 'netF.txt')):
                    print('Write network configs into text')
                    with open(os.path.join(self.save_dir, 'netF.txt'), 'w') as f:
                        print(self.netF, file=f)
                    f.close()
                
                torch.cuda.empty_cache()
                #self.parallelize()

    def optimize_parameters(self, iters):
        # forward
        accum_iter = self.grad_accum_iters #4
        # accumulate 
        with torch.set_grad_enabled(True):
            self.forward()
            self.loss_G = self.compute_G_loss()
            self.loss_G = self.loss_G / accum_iter 
            self.loss_G.backward()

        # weights update
        if ((iters) % accum_iter == 0):
            self.optimizer_G.step()

            if self.opt.lambda_sim > 0 and self.opt.use_mlp and not self.freeze_netF:
                if self.opt.clip_grad:
                    nn.utils.clip_grad_norm_(self.netF.parameters(), max_norm=self.opt.max_norm, norm_type=2,error_if_nonfinite=True)
                self.optimizer_F.step()
            # zero grad 
            self.optimizer_G.zero_grad()
            if self.opt.lambda_sim > 0 and self.opt.use_mlp and not self.freeze_netF:
                self.optimizer_F.zero_grad()

    def set_input(self, input, verbose=False):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.verbose = verbose

        if self.isTrain:
            self.real_A = input['A'].to(self.device).float()  
            if self.opt.lambda_sim > 0:
                self.age_A = input['A_age'].to(self.device).float()
                self.subj_A = input['A_id'].to(self.device).float()
            if self.opt.lambda_Rec > 0:
                self.rec_A_trg = input['A_trg'].to(self.device).float()
            
            if 'B' in input.keys():        
                self.real_B = input['B'].to(self.device).float()
                if self.opt.lambda_sim > 0:
                    self.age_B = input['B_age'].to(self.device).float()
                    self.subj_B = input['B_id'].to(self.device).float()
                    self.label_subjid = torch.cat((self.subj_A, self.subj_B), dim=0).squeeze(-1)
                    self.label_age = torch.cat((self.age_A, self.age_B), dim=0).squeeze(-1)
                if self.opt.lambda_Rec > 0:
                    self.rec_B_trg = input['B_trg'].to(self.device).float()
            
            else:
                self.real_B = None
                if self.opt.lambda_sim > 0:
                    self.label_subjid = self.subj_A.squeeze(-1)
                    self.label_age = self.age_A.squeeze(-1)

            if 'A_seg' in input.keys():
                self.seg_A = input['A_seg'].unsqueeze(1).to(self.device)
                self.visual_names = ['real_A', 'pred_seg_A', 'seg_A']
                self.seg_A_gt = True
            else:
                self.seg_A = None  
                self.seg_A_gt = False

            if 'B_seg' in input.keys():
                self.seg_B = input['B_seg'].unsqueeze(1).to(self.device)
                self.visual_names += ['real_B', 'pred_seg_B', 'seg_B']
                self.seg_B_gt = True
            else:
                self.seg_B = None  
                self.seg_B_gt = False
            del input
            print('A',self.real_A.size()) 

        else:
            self.real_A = input['A'].to(self.device).float()
            if 'A_seg' in input.keys():
                self.seg_A = input['A_seg'].unsqueeze(1).to(self.device).float()
            else:
                self.visual_names = ['real_A', 'pred_seg']
           

    def forward(self, verbose=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            # for segmentation
            if self.real_B is not None:
                self.reals = torch.cat((self.real_A, self.real_B), dim=0)
            else:
                self.reals = self.real_A

            if self.opt.lambda_sim > 0:
                segs, self.feat_kq = self.netG(self.reals, self.sim_layers, False, verbose=verbose)
            else:
                segs = self.netG(self.reals, [], False, verbose=verbose)
            
            if self.opt.lambda_Rec > 0:
                self.recs = self.netRec(segs)
                assert self.recs.size() == self.reals.size()
                self.rec_A = self.recs[:self.real_A.size(0)]
                if self.rec_B_trg is not None:
                    self.recon_trg = torch.cat((self.rec_A_trg, self.rec_B_trg), dim=0)
                else:
                    self.recon_trg = self.rec_A_trg 
            self.pred_A = segs[:self.real_A.size(0)]
            self.pred_seg_A = torch.argmax(F.softmax(self.pred_A.detach(), dim=1), dim=1).float().unsqueeze(1)

            if self.real_B is not None:
                self.pred_B = segs[self.real_A.size(0):]
                self.pred_seg_B = torch.argmax(F.softmax(self.pred_B.detach(), dim=1), dim=1).float().unsqueeze(1)


        else:
            # test time: single input
            crop_size = self.opt.crop_size
            if self.opt.ndims == 2:
                self.pred_seg = torch.zeros_like(self.real_A)
                nframes = self.real_A.size(2)
                b,c,w,h,d = self.real_A.size()
                patch = self.real_A.permute(0,2,1,3,4).view(b*w, c, h, d)
                patch_pred_seg = self.netG(patch, [], False)
                _, cseg, _, _ = patch_pred_seg.size()

                patch_pred_seg = patch_pred_seg.view(b,w,cseg,h,d).permute(0,2,1,3,4)
                patch_pred_seg = torch.argmax(F.softmax(patch_pred_seg, dim=1), dim=1).float().unsqueeze(1)
                self.pred_seg = patch_pred_seg

            elif self.opt.ndims == 3:
                if crop_size == -1:
                    patch = self.real_A
                    self.pred_seg = self.netG(patch, [], False)
                    self.pred_seg = torch.argmax(F.softmax(self.pred_seg, dim=1), dim=1).float().unsqueeze(1)


    def compute_G_loss(self, iters=-1):
        """Calculate  Sim loss for the generator"""
        self.loss_G = 0.
        
        # segmentation consistency loss
        if self.real_B is not None and self.opt.lambda_cseg > 0:
            self.loss_CS = 0.5* (self.criterionSegC(self.pred_A, self.pred_seg_B) + self.criterionSegC(self.pred_B, self.pred_seg_A))
            self.loss_G += (self.loss_CS * self.opt.lambda_cseg)
        else:
            self.loss_CS = 0.

        # segmentation gt dice 
        if self.seg_A_gt:
            self.loss_S = self.criterionSeg(self.pred_A, self.seg_A) #.long())
            with torch.no_grad(): # for checking purpose only 
                self.loss_dice = self.criterionSeg_val.dice(self.pred_A.detach(), self.seg_A.detach())
                self.loss_ce = self.criterionSeg_val.ce(self.pred_A.detach(), self.seg_A.detach())
        else:
            self.loss_S = 0.

        if self.seg_B_gt:
            self.loss_S += self.criterionSeg(self.pred_B, self.seg_B) #.long())

        else:
            pass
    
        if self.seg_A_gt or self.seg_B_gt:
            self.loss_G += (self.loss_S * self.opt.lambda_seg)

        if self.opt.lambda_sim > 0:
            self.loss_Sim, self.sim_dict, self.loss_std, self.std_dict, \
            self.loss_cov, self.cov_dict, self.loss_ortho, self.ortho_dict = self.calculate_Sim_loss(self.feat_kq, None, iter, vis=False) #(step!=-1))
            
            self.loss_G += self.loss_Sim * self.opt.lambda_sim
            self.loss_G += self.loss_std
            self.loss_G += self.loss_cov
            self.loss_G += self.reg_O * self.loss_ortho
        else:
            self.loss_Sim = 0.
            self.loss_std, self.loss_cov, self.loss_ortho = 0., 0., 0.

        if self.opt.lambda_Rec > 0:
            assert self.opt.lambda_seg == 0, 'Reconstruction and segmentation conflicts'
            self.loss_Rec = self.calculate_recon_loss() #self.criterionRec(self.recs, self.reals)
            self.loss_G += self.loss_Rec * self.opt.lambda_Rec
            
        else:
            self.loss_Rec = 0.

        return self.loss_G

    def calculate_recon_loss(self):
        return self.criterionRec(self.recs, self.recon_trg)

    def calculate_Sim_loss(self, feat_kq, mask=None, step=-1, vis=False):
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        feat_kq_pool, sample_ids = self.netF(feat_kq, self.opt.num_patches, None, mask, False)# verbose=self.verbose)
        total_nce_loss, total_std_loss, total_cov_loss, total_ortho_loss = 0.0, 0.0, 0.0, 0.0
        layer_wise_dict = OrderedDict()
        layer_wise_std, layer_wise_cov, layer_wise_ortho = OrderedDict(), OrderedDict(), OrderedDict()
        bs = self.real_A.size(0)
        for f_kq, sample_id, crit, sim_layer, nce_w,feat, lambda_std, lambda_cov in zip(feat_kq_pool, 
                  sample_ids, self.criterionSim, self.sim_layers, self.sim_weights, feat_kq, self.reg_V, self.reg_C):
            f_proj, f_pred = f_kq
            z1, z2 = f_proj[:bs], f_proj[bs:] # non-normalized features
            p1, p2 = f_pred[:bs], f_pred[bs:]
            loss = 0.5 * crit(torch.cat((p1, z2.detach()), dim=0), self.label_age, self.label_subjid, sample_id, feat.size()[2:], debug=False) + \
                   0.5 * crit(torch.cat((p2, z1.detach()), dim=0), self.label_age, self.label_subjid, sample_id, feat.size()[2:], debug=False)

            total_nce_loss += (loss * nce_w * self.opt.lambda_sim)
            layer_wise_dict[str(sim_layer)] = loss.cpu().item()

            if lambda_std > 0 or lambda_cov > 0: 
                ntps, num_patches, nc = z1.size()

                x, y = z1.view(ntps*num_patches, nc).float(), z2.view(ntps*num_patches, nc).float()
                x = x - x.mean(dim=0)
                y = y - y.mean(dim=0)
                std_x = torch.sqrt(x.var(dim=0) + 0.0001)  # variance across batch
                std_y = torch.sqrt(y.var(dim=0) + 0.0001)
                std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
                layer_wise_std[str(sim_layer)+'_Var'] = float(std_loss.cpu().item())

                cov_x = (x.T @ x) / (ntps*num_patches - 1)  # covariance of nc dimension 
                cov_y = (y.T @ y) / (ntps*num_patches - 1)
                cov_loss = off_diagonal(cov_x).pow_(2).sum().div(nc) + \
                        off_diagonal(cov_y).pow_(2).sum().div(nc)
                layer_wise_cov[str(sim_layer)+'_Cov'] = float(cov_loss.cpu().item())

                total_std_loss += lambda_std * std_loss * self.opt.lambda_sim
                total_cov_loss += lambda_cov * cov_loss * self.opt.lambda_sim

        if self.reg_O > 0:
            for (eid, did) in self.concat_pairs:
                f_proj_E, _ = feat_kq_pool[eid]
                sample_id_E = sample_ids[eid]
                f_proj_D, _, sample_id_D = self.netF.sample_by_id(feat_kq[did], did, 
                                                                         self.opt.num_patches, sample_id_E)
                ntps, npatches, nc = f_proj_E.size()
                f_proj_D = f_proj_D.view(ntps*npatches, nc)
                f_proj_E = f_proj_E.view(ntps*npatches, nc) #.detach()
                
                f_proj_D = self.netF.l2norm(f_proj_D)
                f_proj_E = self.netF.l2norm(f_proj_E)

                f_proj_E = f_proj_E.view(ntps*npatches, 1, nc)
                f_proj_D = f_proj_D.view(ntps*npatches, nc, 1)
                loss_ortho = torch.bmm(f_proj_E, f_proj_D)
                loss_ortho = torch.mean(torch.square(loss_ortho.view(-1))) 
                total_ortho_loss += loss_ortho
                layer_wise_ortho['{}_{}_Ortho'.format(eid, did)] = float(loss_ortho.item())
        else:
            total_ortho_loss = 0    
        return total_nce_loss, layer_wise_dict, total_std_loss, layer_wise_std, total_cov_loss, layer_wise_cov, total_ortho_loss, layer_wise_ortho





    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tmp = getattr(self, name)
                if tmp is not None:
                    print(name, tmp.size(), tmp.min().item(), tmp.max().item())
                    visual_ret[name] = tmp
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        if self.opt.lambda_sim > 0:
            errors_ret.update(self.sim_dict)
            errors_ret.update(self.std_dict)
            errors_ret.update(self.cov_dict)
            errors_ret.update(self.ortho_dict)

        return errors_ret


    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
                Norm = networks.get_norm_layer(self.opt.ndims, self.opt.normG)
                if name == 'G' and len(self.unfreeze_layers) > 0:
                    for layer_id, layer in enumerate(net.model):
                        if str(layer_id) not in self.unfreeze_layers and isinstance(layer, Norm):
                            print('Freezing BN stats {}'.format(layer_id))
                            layer.eval()

                if name == 'F' and self.freeze_netF:
                    self.set_requires_grad(net, False)
                    normF = networks.get_norm_layer(1, 'batch')

                    for feat_id, _ in enumerate(self.sim_layers):
                        if isinstance(self.netF, torch.nn.DataParallel):
                            mlp = getattr(self.netF.module, 'mlp_%d' % feat_id)
                            pred = getattr(self.netF.module, 'pred_%d' % feat_id)
                        else:
                            mlp = getattr(self.netF, 'mlp_%d' % feat_id)
                            pred = getattr(self.netF, 'pred_%d' % feat_id)
                        for layer_id, layer in enumerate(mlp):
                            if isinstance(layer, normF):
                                print('[MLP {}] Freezing BN stats {}'.format(feat_id, layer_id))
                                layer.eval()
                        
                        for layer_id, layer in enumerate(pred):
                            if isinstance(layer, normF):
                                print('[Pred {}] Freezing BN stats {}'.format(feat_id, layer_id))
                                layer.eval()
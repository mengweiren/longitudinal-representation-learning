# Reference:
# CUT (https://github.com/taesungp/contrastive-unpaired-translation)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np

##################
# Building blocks
##################
class ConvBlock(nn.Module):
    def __init__(self, ndims, input_dim, output_dim, kernel_size, stride, bias,
                 padding=0, norm='none', activation='relu', pad_type='zeros'):
        super(ConvBlock, self).__init__()
        self.use_bias = bias
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        Conv = getattr(nn, 'Conv%dd' % ndims)
        # initialize padding
        #if pad_type == 'reflect':
        #    self.pad = getattr(nn, 'ReflectionPad%dd'%ndims)(padding)
        #elif pad_type == 'zero':
        #    self.pad = getattr(nn, 'ZeroPad%dd'%ndims)(padding)
        #else:
        #    assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize convolution
        self.conv = Conv(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, padding=padding, padding_mode=pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = getattr(nn, 'BatchNorm%dd'%ndims)(norm_dim)
        elif norm == 'instance':
            self.norm = getattr(nn, 'InstanceNorm%dd'%ndims)(norm_dim, track_running_stats=False)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def get_norm_layer(ndims, norm='batch'):
    if norm == 'batch':
        Norm = getattr(nn, 'BatchNorm%dd' % ndims)
    elif norm == 'instance':
        Norm = getattr(nn, 'InstanceNorm%dd' % ndims)
    elif norm == 'none':
        Norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return Norm

def get_actvn_layer(activation='relu'):
    if activation == 'relu':
        Activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        Activation = nn.LeakyReLU(0.3, inplace=True)
    elif activation == 'elu':
        Activation = nn.ELU()
    elif activation == 'prelu':
        Activation = nn.PReLU()
    elif activation == 'selu':
        Activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
        Activation = nn.Tanh()
    elif activation == 'none':
        Activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return Activation

################
# Generators
################
class UnetGeneratorExpand(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, dimension, input_nc, output_nc, num_downs, ngf=24, norm='batch',
                 final_act='none', activation='relu', pad_type='reflect', doubleconv=True, residual_connection=False, 
                 pooling='Max', interp='nearest', use_skip_connection=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            pooling         -- Max or Avg
            interp          -- 'nearest' or 'trilinear'
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGeneratorExpand, self).__init__()
        # construct unet structure
        ndims = dimension
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        use_bias = norm == 'instance'
        self.use_bias = use_bias

        Conv = getattr(nn, 'Conv%dd' % ndims)
        Pool = getattr(nn, '%sPool%dd' % (pooling,ndims))

        # initialize normalization
        Norm = get_norm_layer(ndims, norm)
        Activation = get_actvn_layer(activation)
        FinalActivation = get_actvn_layer(final_act)

        self.residual_connection = residual_connection
        self.res_dest = []
        self.res_source  = []

        model = [Conv(input_nc, ngf, 3, stride=1, bias=use_bias, padding='same', padding_mode=pad_type)]
        self.res_source += [len(model)-1]
        if Norm is not None:
            model += [Norm(ngf)]
            
        if Activation is not None:
            model += [Activation]
        self.res_dest += [len(model) - 1]

        # Encoder: downsampling blocks
        self.use_skip_connection = use_skip_connection
        self.encoder_idx = []
        in_ngf = ngf
        for i in range(num_downs):
            if i == 0:
                mult = 1
            else:
                mult = 2
            model += [Conv(in_ngf, in_ngf * mult, kernel_size=3, stride=1, bias=use_bias, padding='same',
                           padding_mode=pad_type)]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(in_ngf * mult)]

            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

            if doubleconv:
                model += [Conv(in_ngf * mult, in_ngf * mult, kernel_size=3, stride=1, bias=use_bias, padding='same',
                               padding_mode=pad_type)]
                self.res_source += [len(model) - 1]
                if Norm is not None:
                    model += [Norm(in_ngf * mult)]
                if Activation is not None:
                    model += [Activation]
                self.res_dest += [len(model) - 1]

            self.encoder_idx += [len(model) - 1]
            model += [Pool(2)]
            in_ngf = in_ngf * mult


        model += [
            Conv(in_ngf, in_ngf * 2, kernel_size=3, stride=1, bias=use_bias, padding='same', padding_mode=pad_type)]
        self.res_source += [len(model) - 1]
        if Norm is not None:
            model += [Norm(in_ngf * 2)]

        if Activation is not None:
            model += [Activation]
        self.res_dest += [len(model) - 1]

        if doubleconv:
            #self.conv_id += [len(model)]
            model += [
                Conv(in_ngf * 2, in_ngf * 2, kernel_size=3, stride=1, bias=use_bias, padding='same', padding_mode=pad_type)]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(in_ngf * 2)]
    
            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

        # Decoder
        self.decoder_idx = []
        mult = 2 ** (num_downs)
        for i in range(num_downs):
            self.decoder_idx += [len(model)]
            model += [nn.Upsample(scale_factor=2, mode=interp)]
            if self.use_skip_connection:  # concate encoder/decoder feature
                m = mult + mult // 2
            else:
                m = mult
            #self.conv_id += [len(model)]
            model += [Conv(ngf * m, ngf * (mult // 2), kernel_size=3, stride=1, bias=use_bias,
                           padding='same', padding_mode=pad_type)]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(ngf * (mult // 2))]
            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

            if doubleconv:
                #self.conv_id += [len(model)]
                model += [Conv(ngf * (mult // 2), ngf * (mult // 2), kernel_size=3, stride=1, bias=use_bias, padding='same',
                               padding_mode=pad_type)]
                self.res_source += [len(model) - 1]
                if Norm is not None:
                    model += [Norm(ngf * (mult // 2))]
   
                if Activation is not None:
                    model += [Activation]
                self.res_dest += [len(model) - 1]

            mult = mult // 2

        print('Encoder skip connect id', self.encoder_idx)
        print('Decoder skip connect id', self.decoder_idx)

        Conv = getattr(nn, 'Conv%dd' % ndims)  # no weight standardization at output layer 
        # final conv w/o normalization layer
        model += [
            Conv(ngf * mult, output_nc, kernel_size=3, stride=1, bias=use_bias, padding='same', padding_mode=pad_type)]
        if FinalActivation is not None:
            model += [FinalActivation]
        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False, verbose=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            enc_feats = []
            feat_tmp = dict()
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                #print(layer_id, layer)
                
                if verbose:
                    print(layer_id, layer.__class__.__name__, feat.size())

                if self.residual_connection and layer_id in self.res_source:
                    feat_tmp = feat
                    if verbose:
                        print('Record skip connection input from %d'%layer_id)

                if self.residual_connection and layer_id in self.res_dest:
                    assert feat_tmp.size() == feat.size()
                    feat = feat + 0.1 * feat_tmp
                    if verbose:
                        print('Add skip connection input for %d'%layer_id)

                if self.use_skip_connection:  # use encoder/decoder concat 
                    if layer_id in self.encoder_idx:
                        enc_feats.append(feat)
                    if layer_id in self.decoder_idx:
                        feat = torch.cat((enc_feats.pop(), feat), dim=1)


                if layer_id in layers:
                    if verbose:
                        print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)), feat.size())
                    feats.append(feat)
                else:
                    if verbose:
                        print("%d: skipping %s" % (layer_id, layer.__class__.__name__), feat.size())
                    pass
                if layer_id == layers[-1] and encode_only:
                    if verbose:
                        print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            enc_feats = []
            feat = input
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer.__class__.__name__)
                feat = layer(feat)
                if self.residual_connection and layer_id in self.res_source:
                    feat_tmp = feat
                if self.residual_connection and layer_id in self.res_dest:
                    assert feat_tmp.size() == feat.size()
                    feat = feat + 0.1 * feat_tmp
                
                if self.use_skip_connection:
                    if layer_id in self.decoder_idx:
                        feat = torch.cat((enc_feats.pop(), feat), dim=1)
                    if layer_id in self.encoder_idx:
                        enc_feats.append(feat)
            return feat



def define_G(dimension, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, gpu_ids=[], opt=None, final_act='none', activation='relu',
             pooling='Max', interp='nearest'):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None

    if netG == 'unet':
        # UnetGeneratorExpand uses one list for all layers, which is more flexible for PatchSim index
        net = UnetGeneratorExpand(dimension, input_nc, output_nc, num_downs=4, ngf=ngf, norm=norm, activation=activation,
                                  final_act=final_act, pad_type='reflect', doubleconv=True, residual_connection=False, use_skip_connection=True,
                                  pooling=pooling, interp=interp)
    elif netG == 'conv':
        net = SingleConv(dimension, input_nc, output_nc, ngf, norm=norm, final_act='none', pad_type='reflect')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=True)


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, 
             n_mlps=2, gpu_ids=[], opt=None, activation='relu', arch=2, use_mlp=True):
    if netF == 'simsiam_mlp_sample':
        net = SimSiamPatchSampleF(use_mlp=use_mlp, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc, activation=activation, arch=arch)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)

class SingleConv(nn.Module):
    def __init__(self, dimension, input_nc, output_nc, ngf=24, norm='batch',
                 final_act='none', activation='elu', pad_type='reflect'):
        super(SingleConv, self).__init__()
        use_bias = norm == 'instance'
        self.model = [ConvBlock(dimension, input_nc, output_nc, stride=1, kernel_size=3, bias=use_bias,
                                padding='same', norm='none', activation=final_act, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        #print('Input before normalization: ', x.min().item(), x.max().item())
        assert len(x.size()) == 2, 'wrong shape {} for L2-Norm'.format(x.size())
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

# global pooling + l2 norm
class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class SimSiamPatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[], n_mlps=2, position=False, activation='relu',arch=2):
        # use the same patch_ids for multiple images in the batch
        super(SimSiamPatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        print('Use MLP: {}'.format(use_mlp))
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.n_mlps = n_mlps
        self.position = position
        self.activation = activation
        self.arch=arch

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            norm = get_norm_layer(1, 'batch')
            Activation = get_actvn_layer(self.activation)

            if self.arch == 2: # turn off bias 
                mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc, bias=False), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc, bias=False), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc, bias=False), norm(self.nc, affine=False)])
                pred = nn.Sequential(*[nn.Linear(self.nc, self.nc//4, bias=False), norm(self.nc//4), Activation,
                                        nn.Linear(self.nc//4, self.nc)])
                
            elif self.arch==1:
                mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc), norm(self.nc), Activation,
                                        nn.Linear(self.nc, self.nc), norm(self.nc)])
                pred = nn.Sequential(*[nn.Linear(self.nc, self.nc//4), norm(self.nc//4), Activation,
                                        nn.Linear(self.nc//4, self.nc)])
                

            if self.gpu_ids == "tpu":
                mlp.to("xla:1")
                pred.to("xla:1")
            else:
                if len(self.gpu_ids) > 0 :
                    mlp.cuda()
                    pred.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
            setattr(self, 'pred_%d' % mlp_id, pred)

            print('mlp_%d created, input nc %d'%(mlp_id, input_nc))
            print('pred_%d created, bottleneck nc %d'%(mlp_id, self.nc//4))

        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def sample_by_id(self, feat, feat_id, num_patches, patch_id= None, mask_i=None, verbose=False):
        ndims = len(feat.size()[2:])
        if num_patches > 0:
            if patch_id is not None:  # sample based on given index
                coords = patch_id
                # patch_id is a torch tensor (share idx across batch)
                if ndims == 3:
                    x_sample = feat[:,:,patch_id[:,0],patch_id[:,1],patch_id[:,2]]
                elif ndims == 2:
                    x_sample = feat[:,:,patch_id[:,0],patch_id[:,1]]
                else:
                    raise NotImplementedError
                if verbose:
                    print('Sample basd on given {} idx w/o mask: sample shape: {}'.format(len(patch_id), x_sample.size()))
            else: # sample patch index
                fg_coords = torch.where(mask_i > 0)
                if ndims == 3:
                    (_,_, fg_x, fg_y, fg_z) = fg_coords
                    #print('coords', fg_x.size(), fg_y.size(), fg_z.size())
                elif ndims == 2:
                    (_,_, fg_x, fg_y) = fg_coords
                else:
                    raise NotImplementedError

                patch_id = torch.randperm(fg_x.shape[0], device=feat.device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]

                select_x, select_y = fg_x[patch_id], fg_y[patch_id]
                if ndims == 3:
                    select_z = fg_z[patch_id]
                    coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1), select_z.unsqueeze(1)), dim=1)
                    x_sample = feat[:, :, select_x, select_y, select_z]

                elif ndims == 2:
                    coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1)), dim=1)
                    x_sample = feat[:, :, select_x, select_y]

                else:
                    raise NotImplementedError

                if verbose:
                    print('Masked sampling, patch_id: {} sample shape: {}'.format(len(patch_id), x_sample.size()))

        else:
            x_sample = feat
            coords = []

        tps, nc, nsample = x_sample.size()
        x_sample = x_sample.permute(0,2,1).flatten(0,1)  # tps*nsample, nc

        mlp = getattr(self, 'mlp_%d' % feat_id)
        pred = getattr(self, 'pred_%d' % feat_id)

        x_sample = mlp(x_sample)
        x_pred = pred(x_sample)
        x_sample = x_sample.view(tps, nsample, -1)
        x_pred = x_pred.view(tps, nsample, -1)
        if verbose:
            print('MLP + reshape: {}'.format(x_sample.size()))
            print('feature range ', feat_id, x_sample.min().item(), x_sample.max().item())
            print('\n\n')
        return x_sample, x_pred, coords


    def forward(self, feats, num_patches=64, patch_ids=None, mask=None, verbose=False):
        return_ids = []
        return_feats = []

        if verbose:
            print(f'Net F forward pass: # features: {len(feats)}')

        ndims = len(feats[0].size()[2:])
        if mask is not None:
            if verbose:
                print(f'Using foreground mask {mask.size()}')
            masks = [F.interpolate(mask, size=f.size()[2:], mode='nearest') for f in feats]
        else:
            masks = [torch.ones(f.size()[2:]).unsqueeze(0).unsqueeze(0) for f in feats]

        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            if verbose:
                print(feat_id, 'input -> {}'.format(feat.size()))
            if num_patches > 0:
                if patch_ids is not None:  # sample based on given index
                    patch_id = patch_ids[feat_id]
                    # patch_id is a torch tensor (share idx across batch)
                    if ndims == 3:
                        x_sample = feat[:,:,patch_id[:,0],patch_id[:,1],patch_id[:,2]]
                    elif ndims == 2:
                        x_sample = feat[:,:,patch_id[:,0],patch_id[:,1]]
                    else:
                        raise NotImplementedError
                    if verbose:
                        print('Sample basd on given {} idx w/o mask: sample shape: {}'.format(len(patch_id), x_sample.size()))

                else: # sample patch index
                    mask_i = masks[feat_id]
                    fg_coords = torch.where(mask_i > 0)
                    if ndims == 3:
                        (_,_, fg_x, fg_y, fg_z) = fg_coords
                    elif ndims == 2:
                        (_,_, fg_x, fg_y) = fg_coords
                    else:
                        raise NotImplementedError

                    patch_id = torch.randperm(fg_x.shape[0], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]

                    select_x, select_y = fg_x[patch_id], fg_y[patch_id]
                    if ndims == 3:
                        select_z = fg_z[patch_id]
                        coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1), select_z.unsqueeze(1)), dim=1)
                        x_sample = feat[:, :, select_x, select_y, select_z]

                    elif ndims == 2:
                        coords = torch.cat((select_x.unsqueeze(1), select_y.unsqueeze(1)), dim=1)
                        x_sample = feat[:, :, select_x, select_y]

                    else:
                        raise NotImplementedError

                    if verbose:
                        print('Masked sampling, patch_id: {} sample shape: {}'.format(len(patch_id), x_sample.size()))

            else:
                x_sample = feat
                coords = []

            tps, nc, nsample = x_sample.size()
            x_sample = x_sample.permute(0,2,1).flatten(0,1)  # tps*nsample, nc
            #print(x_sample.size())
            return_ids.append(coords)


            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                pred = getattr(self, 'pred_%d' % feat_id)

                x_sample = mlp(x_sample)
                x_pred = pred(x_sample)
                #x_sample = self.l2norm(x_sample)  # moved l2-norm outside
                x_sample = x_sample.view(tps, nsample, -1)
                #x_pred = self.l2norm(x_pred)
                x_sample = x_sample.view(tps, nsample, -1)
                x_pred = x_pred.view(tps, nsample, -1)
            else:
                x_sample = x_sample.view(tps, nsample, -1)
                x_pred = x_sample.view(tps, nsample, -1)
                
            if verbose:
                print('MLP + reshape: {}'.format(x_sample.size()))
                print('feature range ', feat_id, x_sample.min().item(), x_sample.max().item())
                if self.position:
                    print('ff feature range ', position_enc.min(), position_enc.max())
                #print('patch id check', coords[:20])
                print('\n\n')
            return_feats.append((x_sample, x_pred))

        return return_feats, return_ids


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'const_linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'linear':
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=5e-2, total_iters=opt.n_epochs+ opt.n_epochs_decay, last_epoch=-1)
    elif opt.lr_policy == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.99)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)  # 0.1
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=1e-4, patience=5, min_lr=1e-7)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids == "tpu":
        net.to("xla:1")
    else:
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to("cuda")
            # if not amp:
            #net = torch.nn.DataParallel(net, "gpu_ids")  # multi-GPUs for non-AMP training
        #    net = torch.nn.DataParallel(net, "cuda")
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


##############################################################################
# Data parallel wrapper
##############################################################################
class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            print(name)
            return getattr(self.module, name)




class Pairwise_Feature_Similarity(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.l2norm = Normalize(2)

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        print('Initialized pairwise similarity loss')

    def pairwise_cosine_simililarity(self, x, y):
        assert x.size() == y.size(), f'wrong shape {x.size()} and {y.size()}'
        v = self._cosine_similarity(x,y) #(x.unsqueeze(1), y.unsqueeze(2))
       
        return v

    def forward(self, features, labels_age, labels_subjid, labels_coords, coords_range, debug=False):
        '''
        :param feats:  bs*2, num_patches, nc_feature
        :param labels_age: bs*2 -> currently not consider age
        :param labels_subjid: bs*2
        :return:
        '''
        bs2, num_patches, nc = features.size()
        #print(features.size())

        bs = bs2//2
        feat_A = features[:bs, ...].view(bs*num_patches, nc)
        feat_B = features[bs:, ...].view(bs*num_patches, nc)

        #age_A = labels_age[:bs, ...].unsqueeze(1).repeat(1, num_patches).view(-1).float()
        #age_B = labels_age[bs:, ...].unsqueeze(1).repeat(1, num_patches).view(-1).float()

        device = features.device
        #weight = torch.ones(bs*num_patches).to(device)
        sim_pos = self.pairwise_cosine_simililarity(feat_A, feat_B)
        loss = -(sim_pos).sum() / (num_patches*bs)  # negative cosine similarity
        return loss
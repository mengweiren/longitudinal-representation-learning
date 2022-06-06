import os
import numpy as np
import torch
import torch.utils.data
from data.data_utils import normalize_img
from data.base_dataset import BaseDataset
import torchio as tio
import h5py

def process_age(age, amin=42., amax=97.):
    age = (age - amin) / (amax - amin)  # normalize
    if np.isscalar(age):
        return float(age)
    elif isinstance(age, np.ndarray):
        return age.astype(np.float32)
    else:
        raise NotImplementedError


class Longitudinalh5Dataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.folder = opt.dataroot
        self.isTrain = opt.isTrain
        key = 'train' if self.isTrain else 'val'
        self.dimension = opt.data_ndims
        self.h5_long = self.folder + f'/{key}_placeholder_long.hdf5'
        self.normalize = opt.normalize
        self.percentile = 99.99
        self.zero_centered = False
        self.mode = opt.load_mode
        print('----------- %dd Data Loader Initialization, load mode [%s] ----------'%(self.dimension, self.mode))
        if self.mode == 'long_pairs':  # longitudinal pairs
            print(self.h5_long, os.path.exists(self.h5_long))
            self.hf = h5py.File(self.h5_long, 'r')
            self.subj_id = list(self.hf.keys())
            self.len = len(self.hf.keys())
            print('load tp in order: {}'.format(self.opt.tp_order))

        elif self.mode == 'single_seg':  # one image with its segmentation, for training segmentation net
            self.use_numpy = False
            if self.isTrain:
                self.partial_train = opt.partial_train

                assert self.partial_train.endswith('.hdf5')
                self.h5_seg = self.folder + f'/{self.partial_train}'
                print(f'Using {self.h5_seg}')
                self.hf = h5py.File(self.h5_seg, 'r')
                if self.dimension == 3:
                    self.len = int(self.hf['img_seg_pair']['t1'].shape[0])  # n, 192, 160, 160
                elif self.dimension == 2:
                    self.nframes = self.hf['img_seg_pair']['t1'].shape[1]  # !!! this is first axis only
                    self.len = int(self.hf['img_seg_pair']['t1'].shape[0] * self.nframes)  # n*192, 160, 160
                else:
                    raise NotImplementedError

            else:
                self.h5_seg = self.folder + f'/{key}_{opt.validation_prefix}.hdf5'
                print(self.h5_seg)
                self.hf = h5py.File(self.h5_seg, 'r')
                if self.dimension == 3:
                    self.len = self.hf['img_seg_pair']['t1'].shape[0]  # 98, 192, 160, 160
                elif self.dimension == 2:
                    self.len = self.hf['img_seg_pair']['t1'].shape[0]* self.hf['img_seg_pair']['t1'].shape[1]  # 98, 192, 160, 160
                    self.nframes = self.hf['img_seg_pair']['t1'].shape[1]
                else:
                    raise NotImplementedError

        else:
            raise NotImplementedError
        self.load_seg = True
        self.crop_size = opt.crop_size
        self.load_recon = opt.lambda_Rec > 0

        if self.opt.resize:
            print('Including resizing in preprocessing: %d'%opt.crop_size)
            assert opt.crop_size > 0, 'Wrong size %d specified for resizing.'%(opt.crop_size)
            self.pre_resize = tio.Resize((opt.crop_size, opt.crop_size, opt.crop_size))
       
        self.apply_same_inten_augment = opt.apply_same_inten_augment
        if (opt.isTrain and opt.augment):
            print('Initializing augmentation ...')

            augment_list_intensity = []
            if opt.inten_augment:
                print('Add in intensity augmentation ')
                if opt.blur: 
                    print('Random Blur')
                    augment_list_intensity += [tio.RandomBlur(p=0.3)]
                if opt.noise:
                    print('Noise')
                    augment_list_intensity += [ tio.RandomNoise(p=0.25)]  
                if opt.bias:
                    print('Bias Field')
                    augment_list_intensity += [tio.RandomBiasField(p=0.2)]   
                if opt.gamma:
                    print('Gamma')
                    augment_list_intensity += [tio.RandomGamma(p=0.3, log_gamma=(-0.3, 0.3))]  
                if opt.motion:
                    print('Motion')
                    augment_list_intensity += [tio.RandomMotion(p=0.1)] 
            self.augment_fn_intensity = tio.Compose(augment_list_intensity)
            
            augment_list_spatial = []
            if opt.geo_augment:
                print('Add in geometric augmentation ')
                augment_list_spatial += [tio.RandomFlip(p=0.5)]
                self.scale= 0.2  
                self.degrees = 10 
                print('affine configs: scale {}, degree {}'.format(self.scale, self.degrees))
                if self.dimension == 3:
                    augment_list_spatial += [tio.RandomAffine(p=0.5, scales=self.scale, degrees=self.degrees)]
                elif self.dimension == 2:
                    augment_list_spatial += [tio.RandomAffine(p=0.5, scales=(self.scale,0,0), degrees=(self.degrees,0,0))]
                else:
                    raise NotImplementedError
            self.augment_fn_spatial = tio.Compose(augment_list_spatial)

        else:
            print('No augmentation.')
            self.augment_fn_intensity, self.augment_fn_spatial = None, None

    def __getitem__(self, item):
        return_dict = dict()

        if self.mode == 'long_pairs': # only support 3D
            while item >= len(self.subj_id):
                item = torch.randint(0, self.len, ()).numpy()
            assert self.dimension == 3, f'Only support 3D data loading in mode {self.mode}'
            subj = self.subj_id[item]
            n_tps_per_subj = self.hf[subj]['t1'].shape[0]

            # restrict the pairs to from neighboring timepoints
            if self.opt.tp_order:
                i = torch.randint(0, n_tps_per_subj-1, ()).numpy()
                j = i+1
            else:
                # restrict the pairs to from random timepoints per subject
                i = torch.randint(0, n_tps_per_subj, ()).numpy()
                j = torch.randint(0, n_tps_per_subj, ()).numpy()
                while j == i:
                    j = torch.randint(0, n_tps_per_subj, ()).numpy()

            img_keys = ['A', 'B']
            if self.load_recon:
                img_keys += ['A_trg','B_trg']

            A_orig = normalize_img(self.hf[subj]['t1'][i], percentile=self.percentile, zero_centered=self.zero_centered)[None, ...]
            return_dict['A_age'] = process_age(self.hf[subj]['age'][i].astype(np.float32))
            return_dict['A_id'] = np.asarray([item])  # dummy variable that contains the subject id

            B_orig = normalize_img(self.hf[subj]['t1'][j], percentile=self.percentile, zero_centered=self.zero_centered)[None, ...]
            return_dict['B_age'] = process_age(self.hf[subj]['age'][j].astype(np.float32))
            return_dict['meta'] = '%s_%.1f_%.1f'%(subj, self.hf[subj]['age'][i], self.hf[subj]['age'][j])
            return_dict['B_id'] = np.asarray([item])  # dummy variable that contains the subject id
            

            if self.opt.augment and self.isTrain:
                A = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(A_orig)))
                B = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(B_orig)))

                if self.opt.resize:
                    A = self.pre_resize(A)
                
                A = self.augment_fn_spatial(A)
                if self.load_recon:
                    return_dict['A_trg'] = A['img'][tio.DATA]
                if self.apply_same_inten_augment:
                    if self.load_recon:
                        geo_transform = A.get_composed_history()
                        return_dict['B_trg'] = geo_transform(B)['img'][tio.DATA]

                    A = self.augment_fn_intensity(A)
                    all_transform = A.get_composed_history()
                    B = all_transform(B)
                
                else:
                    geo_transform = A.get_composed_history()
                    B = geo_transform(B)
                    if self.load_recon:
                        return_dict['B_trg'] = B['img'][tio.DATA]

                    A = self.augment_fn_intensity(A)
                    B = self.augment_fn_intensity(B)
                
                return_dict['A'] = A['img'][tio.DATA]
                return_dict['B'] = B['img'][tio.DATA]
                

            else:
                if self.opt.resize:
                    A = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(A_orig)))
                    A = self.pre_resize(A)
                    A_orig = A['img'][tio.DATA]
                    reproduce_transform = A.get_composed_history()
                    B = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(B_orig)))
                    B = reproduce_transform(B)
                    B_orig = B['img'][tio.DATA]

                return_dict['A'] = A_orig
                return_dict['B'] = B_orig
                if self.load_recon:
                    return_dict['A_trg'] = A_orig
                    return_dict['B_trg'] = B_orig

                if self.load_mask:
                    img_keys = ['A', 'B', 'A_mask', 'B_mask']
                    return_dict['A_mask'] = self.hf[subj]['mask'][i][None,...]
                    return_dict['B_mask'] = self.hf[subj]['mask'][j][None,...]

        elif self.mode == 'single_seg' : # for training
            img_keys = ['A', 'A_seg']

            if self.use_numpy is False:
                #print(item)
                if self.dimension == 3:
                    while item >= self.hf['img_seg_pair']['t1'].shape[0]:
                        item = torch.randint(0, self.len, ()).numpy()
                    A_orig = normalize_img(self.hf['img_seg_pair']['t1'][item], percentile=self.percentile, zero_centered=self.zero_centered)[None, ...]
                    A_seg = self.hf['img_seg_pair']['seg'][item]
                    label_age = process_age(np.asarray(self.hf['img_seg_pair']['age'])[item], amin=42, amax=97)

                elif self.dimension == 2:
                    nitem, nslc = int(item/self.nframes), int(item % self.nframes)
                    #print(nitem, nslc)
                    A_orig = normalize_img(self.hf['img_seg_pair']['t1'][nitem][nslc],
                                           percentile=self.percentile, zero_centered=self.zero_centered)[None, ...][None, ...]
                    if self.isTrain:
                        it = 0
                        while np.sum(A_orig) == 0 and it < 10:  # avoid empty slice in training
                            item = torch.randint(0, self.len, ()).numpy()
                            nitem, nslc = int(item / self.nframes), int(item % self.nframes)
                            A_orig = normalize_img(self.hf['img_seg_pair']['t1'][nitem][nslc],
                                                   percentile=self.percentile, zero_centered=self.zero_centered)[None, ...][None, ...]
                            it += 1

                    A_seg = self.hf['img_seg_pair']['seg'][nitem][nslc][None, ...]

                else:
                    raise NotImplementedError
                return_dict['label_age'] = label_age

            else:
                return_dict['label_age'] = process_age(self.data['age'], amin=42, amax=97)
                if self.dimension == 3:
                    A_orig = normalize_img(self.data['t1'], percentile=self.percentile, zero_centered=self.zero_centered)[None, ...]
                    A_seg = self.data['seg']
                elif self.dimension == 2:
                    A_orig = normalize_img(self.data['t1'][item], percentile=self.percentile, zero_centered=self.zero_centered)[None, ...][None, ...]
                    A_seg = self.data['seg'][item][None,...]
                else:
                    raise NotImplementedError

            if self.opt.augment and self.isTrain:
                A = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(A_orig)),
                                label=tio.LabelMap(tensor=torch.from_numpy(A_seg).unsqueeze(0)))
                A = self.augment_fn_intensity(A)
                A = self.augment_fn_spatial(A)
                reproduce_transform = A.get_composed_history()
                return_dict['transform'] = str(list(reproduce_transform))
                return_dict['A'] = A['img'][tio.DATA]
                return_dict['A_seg'] = A['label'][tio.DATA][0]

            else:
                return_dict['A'] = A_orig
                return_dict['A_seg'] = A_seg

            if self.dimension == 2:
                for k in img_keys:
                    return_dict[k] = return_dict[k][0]  # remove the 1st dimension


        else:
            raise NotImplementedError

        return_dict['keys'] = img_keys


        if self.crop_size > 0 and self.opt.isTrain and (not self.opt.resize):
            if self.dimension == 3:
                sx, sy, sz = return_dict['A'].shape[1:]
                crange = self.crop_size // 2

                cx = np.random.randint(crange, sx - crange) if sx > 2*crange else crange
                cy = np.random.randint(crange, sy - crange) if sy > 2*crange else crange
                cz = np.random.randint(crange, sz - crange) if sz > 2*crange else crange
                #cx, cy, cz = 80, 80, 80
                #print(sx, sy, sz, cx, cy, cz)
                for k in img_keys:
                    tmp = return_dict[k]
                    if len(tmp.shape) == 4:
                        return_dict[k] = tmp[:, cx-crange: cx+crange, cy-crange:cy+crange, cz-crange:cz+crange]
                    elif len(tmp.shape) == 3:
                        return_dict[k] = tmp[cx-crange: cx+crange, cy-crange:cy+crange, cz-crange:cz+crange]
                    else:
                        raise NotImplementedError
                del tmp 
            elif self.dimension == 2:
                sx, sy = return_dict['A'].shape[1:]
                crange = self.crop_size // 2

                cx = np.random.randint(crange, sx - crange) if sx > 2 * crange else crange
                cy = np.random.randint(crange, sy - crange) if sy > 2 * crange else crange
                for k in img_keys:
                    tmp = return_dict[k]
                    if len(tmp.shape) == 3:
                        return_dict[k] = tmp[:, cx - crange: cx + crange, cy - crange:cy + crange]
                    elif len(tmp.shape) == 2:
                        return_dict[k] = tmp[cx - crange: cx + crange, cy - crange:cy + crange]
                    else:
                        raise NotImplementedError

            else:
                raise NotImplementedError

        return return_dict

    def __len__(self):
        return max(self.len, self.opt.batch_size)



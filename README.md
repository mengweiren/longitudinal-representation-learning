# Longitudinal Representation Learning
Reference implementation of "*Local Spatiotemporal Representation Learning for Longitudinally-consistent Neuroimage Analysis*"

Given longitudinal neuroimages with scarce annotation, pretrain a image-to-image network (i.e., U-Net) with the proposed self-supervised spatiotemporal representation learning objectives, and finetune with limited annotation (i.e. 1 subject atlas), optionally with a longitudinal consistency-regularization term.

## Prerequisites
Linux (tested with Ubuntu 20.01)

Python 3 (anaconda)

CPU or NVIDIA GPU + CUDA CuDNN

PyTorch (tested with 1.10)

## Dependencies
To install the environment, run 
```shell script
conda env create -f environment.yml 
```

## Data preparation
HDF5 is recommended for faster data loading of large volumetric 3D biomedical images. 
**For Longitudinal pretraining**
As we perform intra-subject sampling, the pretraining data loader requires subject-specific data structure. The h5 file is hierachically indexed by subject-id, and for each subject group, multiple (N) longitudinal acquisitions are stacked as a 4D array of shape [N, w, h, d]. Example code to generate such h5 file is given below:
```python
import h5py
import numpy as np

long_train = h5py.File('./example_dataset/train_long.hdf5','w')
grp = long_train.create_group('subj1')

place_holder_t1=np.random.rand(3,128,160,160)  # 3 timepoints for subj1 
long_train['subj1']['t1']=place_holder_t1

place_holder_age=np.random.rand(3,1)
long_train['subj1']['age']=place_holder_age

long_train.close()
```

**For segmetation finetuning** 
The finetuning is done with cross-sectional image/annotation pairs which does not require subject-specific data loading. The h5 file could be generated as image-segmentation pairs as follows:
```python
sup_train = h5py.File('./example_dataset/train_image_seg_3d.hdf5','w')

place_holder_t1=np.random.rand(3,128,160,160)  # 3 total images for supervised training
place_holder_seg = np.random.randint(33, size=place_holder_t1.shape).astype(float)
place_holder_age=np.random.rand(3,1)

grp = sup_train.create_group('img_seg_pair')

grp['t1']=place_holder_t1
grp['seg']=place_holder_seg
grp['age']=palce_holder_age
sup_train.close()

```

## Pretraining
Run the following command to pretrain the model:
```shell
cd src/scripts
bash script_pretrain.sh
```

## Finetuning
**Without consistency regularization**
Make sure the pretrained model name is correctly specified in the script, and run the following command to finetune the model without segmentation consistency loss:
```shell
cd src/scripts
bash script_finetune.sh
```


**With consistency regularization**
```shell
cd src/scripts
bash script_finetune_w_Lcs.sh
```

Note: For both pretraining and finetuning, the number of epochs, learning rate and optimization parameters are dataset specific and should be adjusted accordingly.

## Reference
This code base architecure is highly motivated by [CUT](https://github.com/taesungp/contrastive-unpaired-translation). We refer to [SimSiam](https://github.com/facebookresearch/simsiam) for our projector and predictor implementation, and [VICReg](https://github.com/facebookresearch/vicreg) for the variance and covariance regularization terms.
  

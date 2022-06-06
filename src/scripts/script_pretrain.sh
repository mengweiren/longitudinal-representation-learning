export ckpt_dir="../../checkpoints/pretrain/"
export dataroot="../../example_dataset/"
export name="pr_long_rep"
export n_epochs=20
export n_epochs_decay=200
export ortho=100
export var=0.001
export cov=0.001

export crop_size=32
export bs=1

python  ../mains/train.py \
       --checkpoints_dir ${ckpt_dir} \
       --name ${name} \
       --dataroot ${dataroot} \
       --dataset_mode longitudinalh5 \
       --model longrep \
       --ndims 3 \
       --input_nc 1 \
       --output_nc 33 \
       --ngf 16 \
       --freeze_bn False \
       --netF simsiam_mlp_sample \
       --n_mlps 3 \
       --lambda_seg 0 \
       --lambda_cseg 0 \
       --num_threads 1 \
       --lr 2e-4 \
       --print_freq 100 \
       --display_ncols 2 \
       --display_slice 64 \
       --display_freq 100 \
       --save_latest_freq 400 \
       --save_freq 4000 \
       --evaluation_freq 200 \
       --load_mode long_pairs \
       --num_patches 128 \
       --crop_size ${crop_size} \
       --batch_size ${bs} \
       --lr_policy const_linear \
       --init_type kaiming \
       --n_val_during_train 25 \
       --n_epochs ${n_epochs} \
       --n_epochs_decay ${n_epochs_decay}  \
       --h5_prefix _mini_ \
       --lambda_Rec 10 \
       --lambda_sim 1 \
       --netF_nc 2048 \
       --normG batch \
       --netG unet \
       --reg_O ${ortho} \
       --grad_accum_iters 1 \
       --continue_train False \
       --ortho_idx 0 \
       --gpu_ids 0 \
       --sim_layers 3,10,17,24,27,31,38,45,52 \
       --sim_weights 1,1,1,1,0,1,1,1,1 \
       --reg_V 0,0,0,0,0,0,${var},${var},${var},0 \
       --reg_C 0,0,0,0,0,0,${cov},${cov},${cov},0 \

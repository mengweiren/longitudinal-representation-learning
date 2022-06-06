export ckpt_dir="../../checkpoints/"
export dataroot="../../example_dataset/"
export name="finetune/ft_seg"

export partial_train="train_placeholder_image_seg_3d.hdf5"
export validation="placeholder_image_seg_3d"
export n_epochs=200
export n_epochs_decay=12000
export pretrained_name="pretrain/pr_long_rep"

# dice loss params
export smooth_dr=1e-3
export smooth_nr=0
export batch_dice=False
export include_background=False

# optimizer params
export beta1=0.5
export beta2=0.999
export eps=1e-3
export weight_decay=1e-2

export crop_size=32 #128
export bs=3

python ../mains/train.py \
       --checkpoints_dir ${ckpt_dir} \
       --name ${name} \
       --dataroot ${dataroot} \
       --validation_prefix ${validation} \
       --dataset_mode longitudinalh5 \
       --model longrep \
       --ndims 3 \
       --input_nc 1 \
       --output_nc 33 \
       --ngf 16 \
       --normG batch \
       --netG unet \
       --netF simsiam_mlp_sample \
       --gpu_ids 0 \
       --lambda_sim 0 \
       --lambda_seg 1 \
       --lambda_cseg 0 \
       --lambda_Rec 0 \
       --num_threads 1 \
       --lr 2e-4 \
       --lr_policy const_linear \
       --print_freq 1 \
       --partial_train ${partial_train}\
       --display_ncols 2 \
       --display_slice 64 \
       --display_freq 80 \
       --save_latest_freq 10 \
       --evaluation_freq 80 \
       --save_freq 160 \
       --n_epochs ${n_epochs} \
       --n_epochs_decay ${n_epochs_decay} \
       --load_mode single_seg \
       --crop_size ${crop_size} \
       --batch_size ${bs} \
       --pretrained_name ${pretrained_name} \
       --epoch best_val \
       --lr_policy const_linear \
       --smooth_dr ${smooth_dr} \
       --smooth_nr ${smooth_nr} \
       --batch_dice ${batch_dice} \
       --include_background ${include_background}\
       --beta1 ${beta1} \
       --beta2 ${beta2} \
       --eps ${eps} \
       --weight_decay ${weight_decay} \
       --ams_grad ${ams_grad} \
       --pretrained_name ${pretrained_name} \
       --pool_type Max \
       --interp_type nearest \
       --augment True 
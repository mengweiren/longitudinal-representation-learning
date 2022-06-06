export ckpt_dir="../../checkpoints/"
export dataroot="../../example_dataset/"
export name="finetune/ft_seg_Lcs"

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

export n_epochs=10
export n_epochs_decay=200
export crop_size=128
export bs=3

python ../mains/train_conseg.py \
       --checkpoints_dir ${ckpt_dir} \
       --name ${name} \
       --dataroot ${dataroot} \
       --model longrep \
       --dataset_mode longitudinalh5 \
       --ndims 3 \
       --data_ndims 3 \
       --input_nc 1 \
       --output_nc 33 \
       --ngf 16 \
       --normG batch \
       --netG unet \
       --gpu_ids -1 \
       --lambda_seg 1 \
       --num_threads 1 \
       --lr 2e-4 \
       --print_freq 1 \
       --partial_train ${partial_train} \
       --display_ncols 2 \
       --display_slice 64 \
       --display_freq 80 \
       --save_latest_freq 10 \
       --evaluation_freq 80 \
       --save_freq 2000 \
       --n_epochs ${n_epochs} \
       --n_epochs_decay ${n_epochs_decay} \
       --lr_policy const_linear \
       --load_mode long_pairs \
       --augment True \
       --crop_size ${crop_size} \
       --batch_size ${bs} \
       --init_type kaiming \
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
       --epoch best_val \
       --epoch_count 0 \
       --lambda_sim 0 \
       --lambda_cseg 1 \
       --actG relu \
       --actF relu \
       --lambda_Rec 0 \
       --grad_accum_iters 1 \
       --validation_prefix ${validation}
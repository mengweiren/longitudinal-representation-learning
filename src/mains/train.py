#!/usr/bin/env python

from copy import deepcopy
import os, sys
sys.path.append('../')
import time
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualization import Visualizer
import torchio as tio
from tqdm import tqdm
from util.util import save_tensor
from torch import nn
from glob import glob

# Load experiment setting
opt = TrainOptions().parse()   # get training options

train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
bs = opt.batch_size
crop = opt.crop_size
resize = opt.resize
print('The number of training images = %d' % train_dataset_size)

val_opt = deepcopy(opt)
val_opt.isTrain = False
val_opt.batch_size = 1
val_opt.crop_size = -1

val_dataset = create_dataset(val_opt)
val_dataset_size = len(val_dataset)
print('The number of validation images = %d' % val_dataset_size)
opt.isTrain = True
opt.batch_size = bs
opt.crop_size = crop

print(opt.model)
model = create_model(opt)      # create a model given opt.model and other options
print('* Creating tensorboard summary writer %s'%model.save_dir)
model.visualizer = Visualizer(opt)

total_iters = 0

slc = 80

optimize_time = 0.1

last_eval_loss = -1
best_evaluation_loss = 9999

last_train_loss = -1
cur_train_loss = 99

if opt.continue_train and opt.epoch == 'latest':
    opt.pretrained_name = opt.name
    print('Retrieve latest checkpoints from %s'%model.save_dir)
    x = set(glob(model.save_dir + '/*net_G.pth')) - set(glob(model.save_dir +"/*latest_net_G.pth")) - set(glob(model.save_dir +"/*best_val*"))
    if len(x) > 0:
        latest = sorted(x, key=os.path.getmtime)[-1].split('/')[-1].split('_')[0]
        opt.epoch = latest
        latest_epoch = int(latest)//train_dataset_size
        opt.epoch_count = latest_epoch
        print('Found %d checkpoints, take the latest one %s (iters), latest epoch %d'%(len(x), opt.epoch, latest_epoch))

        with open(model.save_dir + '/best_val_loss.txt','r' ) as f:
            best_evaluation_loss =  float(f.readline().rstrip())
            print('Load evaluation record : %f'%best_evaluation_loss)
        if latest_epoch > opt.epoch_count:
            print('Retrieve learning rate')
            for _ in range(opt.epoch_count, opt.latest_epoch):
                model.update_learning_rate()
            opt.lr = model.optimizers[0].param_groups[0]['lr']
    else:
        print('No checkpoints found, start from scratch')
        opt.continue_train = False
        opt.epoch = 0
        opt.epoch_count = 0

est_iters = int(train_dataset_size/bs) * bs * (opt.n_epochs+opt.n_epochs_decay)
print('Training start from epoch %d, total epochs: %d - est iters: %d'%((opt.epoch_count, opt.n_epochs+opt.n_epochs_decay, est_iters)))
total_iters = opt.epoch_count * train_dataset_size
print('Starting iters: {}'.format(total_iters))


times = []
t_data = 0

for epoch in range(opt.epoch_count,
                   opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    if epoch > opt.stop_epoch:
        print('stop training at epoch %d'%epoch)
        break 

    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    train_dataset.set_epoch(epoch)
    for i, data in enumerate(train_dataset):  # inner loop within one epoch
        batch_size = data["A"].size(0)
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += batch_size
        epoch_iter += batch_size
        if len(opt.gpu_ids) > 0 and opt.gpu_ids != 'tpu':
            torch.cuda.synchronize()
        optimize_start_time = time.time()

        if epoch == opt.epoch_count and i == 0:
            print('Data dependent initialization')
            for  k in set(data['keys'][0]):
                print(k, data[k].size())
            model.data_dependent_initialize(data)
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.train()

        verbose = (total_iters % opt.print_freq == 0)  # print out sample information every 1k iters
        if verbose:
            print(f'Start of iters [{total_iters}]')

        model.set_input(data, verbose)  # unpack data from dataset and apply preprocessing
        model.optimize_parameters(total_iters)  # calculate loss functions, get gradients, update network weights
        if len(opt.gpu_ids) > 0 :
            torch.cuda.synchronize()

        del data
        if len(opt.gpu_ids) > 0:
            torch.cuda.empty_cache()

        optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time
        if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
             model.visualizer.display_current_results(model.get_current_visuals(), total_iters, board_name='train')


        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            model.visualizer.print_current_losses(total_iters, epoch, epoch_iter, losses, optimize_time, t_data)
            model.visualizer.plot_current_losses(total_iters, losses)
            model.visualizer.writer.add_scalar('learning_rate', model.optimizers[0].param_groups[0]['lr'], total_iters)

        if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            print(opt.name)  # it's useful to occasionally show the experiment name on console
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)
            visuals = model.get_current_visuals()  # get image results
            if len(visuals)>0:
                save_tensor('nii', model.save_dir + '/nii_latest/'.format(opt.crop_size), 'train', visuals)


        iter_data_time = time.time()

        if total_iters % opt.evaluation_freq == 0:
            model.save_networks(total_iters)
            model.eval()

            cur_eval_loss = 0
            cur_dice, cur_ce = 0,0
            cnt_data = 0
            with torch.no_grad():
                print(f'{total_iters} iters, Evaluation starts ...')
                for vcnt, data in enumerate(val_dataset):  # inner loop within one epoch
                    if vcnt > opt.n_val_during_train:
                        break
                    model.set_input(data, verbose=True)  # unpack data from dataset and apply preprocessing
                    model.forward()  # calculate loss functions, get gradients, update network weights
                    cur_eval_loss += model.compute_G_loss(-1).item()
                    cnt_data += 1
                    if opt.lambda_seg > 0:
                        cur_dice += model.loss_dice.item()
                        cur_ce += model.loss_ce.item()
              
                del data
                if len(opt.gpu_ids) > 0  and opt.gpu_ids != 'tpu':
                    torch.cuda.empty_cache()
                cur_eval_loss = cur_eval_loss/cnt_data
                cur_dice = cur_dice/cnt_data
                cur_ce = cur_ce/cnt_data
                print(f'Best validation loss: {best_evaluation_loss}, current validation loss {cur_eval_loss}')

                model.visualizer.display_current_results(model.get_current_visuals(), total_iters, board_name='val')

                delta = abs(last_eval_loss - cur_eval_loss)

                model.visualizer.plot_current_losses(total_iters, {'current_val': cur_eval_loss})

                if opt.lambda_seg > 0:
                    model.visualizer.plot_current_losses(total_iters, {'current_val_dice': cur_dice, 'current_val_ce': cur_ce})

                if cur_eval_loss < best_evaluation_loss:
                    best_evaluation_loss = cur_eval_loss
                    print(f'Saving model with best validation loss: {best_evaluation_loss}, last validation loss {last_eval_loss}, delta {delta}')
                    model.save_networks('best_val')

                    model.visualizer.plot_current_losses(total_iters, {'best_val': best_evaluation_loss})
                    with open(model.save_dir + '/best_val_loss.txt', 'w') as f:
                        f.writelines(f'{best_evaluation_loss}')
                last_eval_loss = cur_eval_loss

                if opt.lr_policy == 'plateau':
                    model.update_learning_rate(cur_eval_loss)

            model.train()
        if verbose:
            print(f'End of iters [{total_iters}]')

    if total_iters % opt.save_freq == 0:  # cache our model every <save_freq> iterations
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(total_iters)
        visuals = model.get_current_visuals()  # get image results
        if len(visuals)>0:
            save_tensor('nii', model.save_dir + '/nii_latest/'.format(opt.crop_size), 'train', visuals)

    print('Total iters %d, End of epoch %d / %d \t Time Taken: %d sec' % (
    total_iters, epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    if not opt.lr_policy == 'plateau':
        model.update_learning_rate()  # update learning rates at the end of every epoch.


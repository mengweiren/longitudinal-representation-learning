import argparse
import os
from posixpath import pardir
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU, -2 for TPU')

        parser.add_argument('--checkpoints_dir', type=str, default='../../checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='longcl', help='chooses which model to use.')
        parser.add_argument('--ndims', type=int, default=3, help='network dimension: 2|3')
        parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=33, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in the last conv layer')
        parser.add_argument('--skip_connection', type=util.str2bool, nargs='?',const=True,default=False,help='whether to add skip connection')
        parser.add_argument('--netG', type=str, default='unet', choices=['noiseunet','noskipunet','unet'], help='specify generator architecture')
        parser.add_argument('--normG', type=str, default='batch', choices=['instance', 'batch', 'none','layer'], help='instance normalization or batch normalization for G')
        parser.add_argument('--actG', type=str, default='relu', choices=['relu', 'lrelu'], help='relu or lrelu')
        parser.add_argument('--actF', type=str, default='relu', choices=['relu', 'lrelu'], help='relu or lrelu')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
        parser.add_argument('--pool_type', type=str, default='Max', choices=['Max', 'Avg'], help='pooling type for downsampling')
        parser.add_argument('--interp_type', type=str, default='nearest', choices=['nearest', 'trilinear'], help='interpolation type for upsampling')
        parser.add_argument('--lambda_Rec', type=float, default=0., help='weight for Recon loss')

        # dice loss parameters
        parser.add_argument('--smooth_dr', type=float, default=1e-5, help='denominator smoothing of Dice')
        parser.add_argument('--smooth_nr', type=float, default=1e-5, help='numerator smoothing of Dice')
        parser.add_argument('--batch_dice', type=util.str2bool, nargs='?', const=True, default=False, help='Whether to use batch dice')
        parser.add_argument('--include_background', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to include background in dice')


        # dataset parameters
        parser.add_argument('--dataroot', default='placeholder', help='path to images')
        parser.add_argument('--dataset_mode', type=str, default='longitudinalIBIS', help='chooses datasets')
        parser.add_argument('--validation_prefix', type=str, default='image_seg_3d', help='chooses validation file')
        parser.add_argument('--data_ndims', type=int, default=3, help='data dimension')
        parser.add_argument('--h5_prefix', type=str, default='_', choices=['_mini_','_'], help='prefix of h5 for getting different settings')
        
        parser.add_argument('--tp_order', type=util.str2bool, nargs='?', const=True, default=False, help='Whether to load tps in order')
        parser.add_argument('--load_mode', type=str, default='random', help='load subject&tps randomly (train) or by order (test time)')
        parser.add_argument('--id_age_list', type=str, default='txt file placeholder', help='subject based text file for training')
        parser.add_argument('--normalize', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to normalize data')

        # augmentation specs
        parser.add_argument('--augment', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to augment data')
        parser.add_argument('--geo_augment', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to perform geometric augmentation')
        parser.add_argument('--inten_augment', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to perform intensity augmentation')
        parser.add_argument('--apply_same_inten_augment', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to perform the same intensity augmentation on timepoint1 & 2')
        parser.add_argument('--blur', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to add augment [blur]')
        parser.add_argument('--noise', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to add augment [noise]')
        parser.add_argument('--bias', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to add augment [bias]')
        parser.add_argument('--gamma', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to add augment [gamma]')
        parser.add_argument('--motion', type=util.str2bool, nargs='?', const=True, default=True, help='Whether to add augment [motion]')

        parser.add_argument('--resize', type=util.str2bool, nargs='?', const=True, default=False, help='Whether to resize data, if true, no crop')

        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--batch_size2', type=int, default=3, help='input batch size for Lcs')
        parser.add_argument('--crop_size', type=int, default=128, help='crop size')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))

        if os.path.exists(file_name):
            import datetime
            file_name = file_name.replace('opt.txt', 'opt_{}.txt'.format(str(datetime.datetime.now())[:10]))
        try:
            with open(file_name, 'w') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        if opt.gpu_ids != 'tpu':
            # set gpu ids
            str_ids = opt.gpu_ids.split(',')
            opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    opt.gpu_ids.append(id)
            if len(opt.gpu_ids) > 0:
                torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

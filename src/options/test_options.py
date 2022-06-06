from random import choice
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=1000, help='maximum number of images')
        parser.add_argument('--j', type=int, default=1, help='parallel running')
        parser.add_argument('--average', action='store_true', help='only used when crop size is not -1, if True, average sliding window results')
        parser.add_argument('--save_name', type=str, default='val',help='txt file prefix')

        parser.add_argument('--evaluation_space', type=str, default='affine', choices=['raw','affine','affine_and_deform'], help='evaluate in which space')
        parser.add_argument('--evaluation_data', type=str, default='prediction', choices=['prediction','static'], help='evaluate on data')

        self.isTrain = False
        return parser

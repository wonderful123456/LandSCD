import argparse
import torch
import os

class ParserOptions():
    """This class defines options that are used by the program"""

    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Semantic Video Segmentation training')

        parser.add_argument('--model', type=str, default='DANet', choices=['deeplab', 'deeplab-50', 'unet', 'unet_paper', 'unet_pytorch', 'pspnet'], help='model name (default: deeplab)')
        parser.add_argument('--mode', type=str, default='fbf-previous', choices=['fbf', 'fbf-1234', 'fbf-previous', 'sequence-1234'], help='training type (default: frame by frame)')
        parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop', 'amsgrad', 'adabound'])
        parser.add_argument('--clip', type=float, default=5, help='gradient clip, 0 means no clip (default: 0)')
        parser.add_argument('--debug', type=int, default=1)
        parser.add_argument('--results_dir', type=str, default='./results_final_clock', help='models are saved here')
        parser.add_argument('--save_dir', type=str, default='saved_models')

        parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes'], help='dataset (default:cityscapes)')
        parser.add_argument('--resize', type=str, default=2, help='image resize: h,w')
        parser.add_argument('--base_size', type=int, help='image h', default=256)
        parser.add_argument('--use_class_weights', type=int, default=0, metavar='N', help='use class weights for the loss function')
        parser.add_argument('--weighting_mode', default='enet', choices=['enet', 'median-freq', 'github-found'])
        parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='input batch size for training (default: 2)')

        parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
        parser.add_argument('--gamma', type=float, default=1, help='gamma value for focal loss')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: auto)')
        parser.add_argument('--lr_policy', type=str, default='poly', choices=['poly', 'step', 'cos', 'linear'], help='lr scheduler mode: (default: poly)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--norm_layer', type=str, default='batch', choices=['instance', 'batch', 'syncbn'])
        parser.add_argument('--init_type', type=str, default='normal', choices=['normal', 'kaiming', 'xavier', 'orthogonal'])

        parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='starting epoch')
        parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: auto)')
        parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
        parser.add_argument('--trainval', type=int, default=0, choices=[0,1], help='determines whether whe should use validation images as well for training')
        parser.add_argument('--vis_split', type=str, default='val', choices=['val', 'test', 'demoVideo'], help='split on which we should produce visualization images')
        parser.add_argument('--submit_format', type=int, default=0, choices=[0,1], help='if set to 1 we save images in the format required for submission in cityscapes else color images')

        # deeplab specific
        parser.add_argument('--pretrained_resnet', type=int, default=0, choices=[0,1], help="if we should use pretrained resnet or not" )
        parser.add_argument('--output_stride', type=int, default=16, help='network output stride (default: 16)')

        # unet specific
        parser.add_argument('--num_downs', type=int, default=9, help='number of unet encoder-decoder blocks')
        parser.add_argument('--ngf', type=int, default=128, help='# of gen filters in the last conv layer')
        parser.add_argument('--remove_skip', type=int, default=0, help='if skip connections should be removed from the model')
        parser.add_argument('--down_type', type=str, default='maxpool', choices=['strideconv, maxpool'], help='method to reduce feature map size')
        parser.add_argument('--dropout', type=float, default=0.2)

        # sequence specific
        parser.add_argument('--sequence_stacked_models', type=int, default=1, help='number of stacked sequence models')
        parser.add_argument('--sequence_model', type=str, default='lstm', choices=['none', 'lstm', 'gru', 'convlstm', 'tcn', 'tcn2d', 'tcn2dhw',
                                                                                   'tcn+temporal_skip', 'tcn+temporal_encoder', 'tcn+temporal_all',
                                                                                   'tcn2d+temporal_skip', 'tcn2d+temporal_encoder', 'tcn2d+temporal_all',
                                                                                   'tcn2dhw+temporal_skip', 'tcn2dhw+temporal_encoder', 'tcn2dhw+temporal_all',
                                                                                   'convlstm+temporal_skip', 'convlstm+temporal_encoder',  'convlstm+temporal_all'])

        parser.add_argument('--shuffle', type=int, default=0, help='if we should shuffle sequence images')
        parser.add_argument('--timesteps', type=int, default=4)
        parser.add_argument('--blank', type=int, default=0, choices=[0,1], help='if we should blank the 4th frame or not (default: 0)')
        parser.add_argument('--time_dilation', type=int, default=2, metavar='N', help='frames -3*t, -2*t, -1*t and 0 will be considered as input frames for GT')

        # lstm specific
        parser.add_argument('--lstm_learning_rate', type=float, help='learning rate (default: auto)')
        parser.add_argument('--lstm_bidirectional', type=bool, default=False, help='if we should use bidirectional lstm or not')
        parser.add_argument('--lstm_initial_state', type=str, default='cnn-learned', choices=['0', '0-learned', 'cnn-learned'])

        # tcn specific
        parser.add_argument('--num_levels_tcn', type=int, default=1, help='number of levels for tcn')
        parser.add_argument('--tcn_kernel_size', type=int, default=3, help='kernel size of tcn')

        # segmentation specific
        parser.add_argument('--segmentation', type=int, default=1 , choices=[0,1], help='segment image based on given mode')

        # reconstruction specific
        parser.add_argument('--reconstruct', type=int, default=0, choices=[0,1], help='reconstruct future image')
        parser.add_argument('--reconstruct_loss_type', type=str, default='mse', choices=['mse', 'bce', 'bce-logit', 'ssim'])
        parser.add_argument('--reconstruct_loss_coeff', type=float, default=1, help='the coefficient with which the reconstruction loss will be modified')
        parser.add_argument('--reconstruct_remove_skip', type=int, default=0, choices=[0,1], help='if we should remvoe skip connections in the reconstruction head')

        args = parser.parse_args()

        if not 'sequence' in args.mode:
            args.sequence_model = ''
            args.timesteps = 1
            args.shuffle = 0
            args.clip = 0

        if 'tcn' in args.sequence_model and not 'tcn2d' in args.sequence_model:
            args.tcn_kernel_size = 1

        if 'tcn2dhw' in args.sequence_model:
            args.tcn_kernel_size = 1
            args.num_levels_tcn = 1

        if args.sequence_model != 'lstm':
            args.lstm_initial_state = None

        if args.debug:
            args.results_dir = './results_dummy'

        if args.shuffle:
            args.results_dir = './results_shuffle'

        args.resize = tuple([int(x) for x in args.resize.split(',')])
        args.cuda = torch.cuda.is_available()
        args.gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'] if ('CUDA_VISIBLE_DEVICES' in os.environ) else ''
        args.gpu_ids = list(range(len(args.gpu_ids.split(',')))) if (',' in args.gpu_ids and args.cuda) else None

        if args.gpu_ids and args.norm_layer == 'batch':
            args.norm_layer = 'syncbn'

        if not args.reconstruct and not args.segmentation:
            raise Exception('Both segmentation and reconstruction are disabled')

        self.args = args

    def parse(self):
        return self.args

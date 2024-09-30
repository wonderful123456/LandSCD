# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from argparse import ArgumentParser
# from trainer import CDTrainer
from configs.GEPNew.parser_options import ParserOptions
# from util.general_functions import print_training_info
# from dataloader.Sentinel_Datasets import Sentinel
from dataset.CD_dataset import CDDataset

import torch
from torch.utils.data import DataLoader
import os
# from misc.metric_tool import *
# from misc.logger_tool import *
# import torch.optim as optim
from trainer import CDTrainer
# import utils

def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
def train(args):
    torch.cuda.empty_cache()
    train_sets = CDDataset(split='train')
    # train_sets_aug = Sentinel(split='train')
    # train_sets_aug_aug = Sentinel(split='train')

    # train_sets = torch.utils.data.ConcatDataset([train_sets, train_sets_aug])
    # train_sets = torch.utils.data.ConcatDataset([train_sets, train_sets_aug_aug])
    val_sets = CDDataset(split='val')
    # val_sets_aug = Sentinel(split='val')
    # # val_sets_aug_aug = Sentinel(split='val')
    # val_sets = torch.utils.data.ConcatDataset([val_sets, val_sets_aug])
    # val_sets = torch.utils.data.ConcatDataset([val_sets, val_sets_aug_aug])
    datasets = {'train': train_sets, 'val': val_sets}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, drop_last=True,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    # from models.evaluator import CDEvaluator
    from evaluator import CDEvaluator
    test_sets = CDDataset(split='test')
    # test_sets_aug = Sentinel(split='test')
    dataloader = DataLoader(test_sets, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

    # dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
    #                               batch_size=args.batch_size, is_train=False,
    #                               split='test')

    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()

# def main():

    args = ParserOptions().parse()  # get training options
    # trainer = Trainer(args)
    #
    # print_training_info(args)
    #
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #
    #     if epoch % args.eval_interval == (args.eval_interval - 1):
    #         trainer.validation(epoch)
    #
    # if args.segmentation:
    #     trainer.visualization(args.vis_split)
    #     trainer.save_network()
    #
    # trainer.summary.writer.add_scalar('val/bestmIoU', trainer.best_mIoU, args.epochs)
    # trainer.summary.writer.close()

if __name__ == "__main__":
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--train_model', default='DANet', type=str)
    parser.add_argument('--project_name', default='DANet', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints/', type=str)

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='GEP', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=6, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--num_expert', default=1, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--max_epochs', default=300, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=300, type=int)

    args = parser.parse_args()

    
    get_device(args)
    # print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)

    test(args)

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

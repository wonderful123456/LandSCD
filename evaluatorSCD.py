import os
import numpy as np
import matplotlib.pyplot as plt

# from models.network import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
# from utils import de_norm
# import utils
import torch
from models.danet import get_danet


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = get_danet()
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        # print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.G_pred_B = None
        self.G_pred_CH = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _update_metric(self):
        """
        update metric
        """
        # target = self.batch['L'].to(self.device).detach()
        target = self.batch[1].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        # B Seg
        target_B = self.batch[3].to(self.device).detach()
        G_pred_B = self.G_pred_B.detach()
        G_pred_B = torch.argmax(G_pred_B, dim=1)

        # Change
        target_CH = self.batch[4].to(self.device).detach()
        G_pred_CH = self.G_pred_CH.detach()
        G_pred_CH = torch.argmax(G_pred_CH, dim=1)
        #
        # target = target.reshape(target.shape[0] * target.shape[1], target.shape[2], target.shape[3])
        # G_pred = G_pred.reshape(G_pred.shape[0] * G_pred.shape[1], G_pred.shape[2], G_pred.shape[3])

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        current_score_B = self.running_metric_B.update_cm(pr=G_pred_B.cpu().numpy(), gt=target_B.cpu().numpy())
        current_score_CH = self.running_metric_change.update_cm(pr=G_pred_CH.cpu().numpy(), gt=target_CH.cpu().numpy())
        return current_score + current_score_B + current_score_CH

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        # if np.mod(self.batch_id, 100) == 1:
        message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                  (self.is_training, self.batch_id, m, running_acc)
        self.logger.write(message)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch

        img = batch[0].to(self.device)
        img_B = batch[2].to(self.device)
        self.G_pred, self.G_pred_B, self.G_pred_CH = self.net_G(img, img_B)[0], self.net_G(img, img_B)[1], \
        self.net_G(img, img_B)[2]

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
            # print(self.batch_id)
            # print('done done')
        self._collect_epoch_states()

    def to(self, device):
        pass

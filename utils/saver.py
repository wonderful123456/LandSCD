import os
import glob
import torch
import numpy as np
import json
from collections import OrderedDict


# result/experiment_id/config + logs/losses+metrics + state/epoch_val_loss_xxx.pth.tar
class Saver(object):
    def __init__(self, args):
        self.args = args
        self.dir = args.result_dir     

        experiments = glob.glob(os.path.join(self.dir, args.congfig_name + '_*'))
        experiments_id = sorted([int(id.split('_')[-1]) for id in experiments])[-1] + 1 if experiments else 1
        self.experiment_dir = os.path.join(self.dir, args.congfig_name + '_' + str(experiments_id))    
        self.log_dir = os.path.join(self.experiment_dir, 'log')
        self.state_dir = os.path.join(self.experiment_dir, 'state')
        self.make_folders()
        self.save_experiment_config()
     
    def make_folders(self):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)           
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)

    def _sanitize_filename(self, filename):
        # 过滤掉非法字符
        valid_chars = "-_.() %s%s" % (chr(97), chr(122))  # a-z A-Z 0-9
        return ''.join(c for c in filename if c in valid_chars)
            
    def save_checkpoint(self, state, epoch, metric, checkpoint_name=None):
        if checkpoint_name is not None:
            safe_name = self._sanitize_filename(checkpoint_name)
            filepath = os.path.join(self.state_dir, safe_name + '_checkpoint.pth.tar')
        else:
            filepath = os.path.join(self.state_dir, 'checkpoint.pth.tar')

        try:
            torch.save(state, filepath)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")


    def save_experiment_config(self):
        configfile = os.path.join(self.experiment_dir, 'configs.json')
        configfile = open(configfile, 'w+')
        p = OrderedDict(vars(self.args))   
        json.dump(dict(p), configfile)
        configfile.close()

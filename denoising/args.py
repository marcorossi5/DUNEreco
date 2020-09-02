import os
from datetime import datetime as dtm

class Args:
    def __init__(self, dir_name, epochs, model,\
                 device, loss_fn, lr=0.009032117010326078, amsgrad=True,\
                 out_name=None, scan=False, batch_size=256,\
                 load_path=None, warmup='dn'):
        self.crop_size = (32,32)
        self.crop_p = 0.99
        self.dev_ids = None

        #argparser
        self.dataset_dir = dir_name
        self.epochs = epochs
        self.warmup_roi_epochs = 10
        self.warmup_dn_epochs = 20
        self.model = model
        self.warmup = warmup
        self.device = device
        self.loss_fn = loss_fn
        self.scan = scan

        self.batch_size = batch_size
        self.test_batch_size = batch_size
        self.num_workers = 8

        #model parameters
        self.a = 0.84
        self.k = 8
        self.in_channels = 1
        self.hidden_channels = 32
        
        self.lr_dn = lr
        self.lr_roi = 1e-3
        self.amsgrad = amsgrad

        #logs
        self.plot_dataset = False
        self.plot_acts = True

        self.epoch_log = 1
        self.epoch_test_start = 0
        self.epoch_test = 5

        self.t = 0.5

        self.load = False if (self.load_path is None) else True
        self.load_epoch = 0
        self.load_path = load_path

        self.save = True
        #self.epoch_save = 5

        #build directories
        if out_name is None:
            t = dtm.now().strftime("%y%m%d_%H%M%S")
            self.dir_output = "./denoising/output/%s"%t
        else:
            self.dir_output = "./denoising/output/%s"%out_name
        self.dir_timings = self.dir_output + "/timings"
        self.dir_testing = self.dir_output + "/testing"
        self.dir_final_test = self.dir_output + "/final_test"
        self.dir_metrics = self.dir_output + "/metrics"
        self.dir_saved_models = self.dir_output + "/model_save"

        if not os.path.isdir(self.dir_output):
            os.makedirs(self.dir_output)
        if not os.path.isdir(self.dir_testing):
            os.mkdir(self.dir_testing)
        if not os.path.isdir(self.dir_final_test):
            os.mkdir(self.dir_final_test)
        if not os.path.isdir(self.dir_metrics):
            os.mkdir(self.dir_metrics)
        if not os.path.isdir(self.dir_saved_models):
            os.mkdir(self.dir_saved_models)
        if not os.path.isdir(self.dir_timings):
            os.mkdir(self.dir_timings)

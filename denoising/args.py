import os
from datetime import datetime as dtm

class Args:
    def __init__(self, dir_name, epochs, model, device, loss_fn, lr, out_name=None):
        self.crop_size = (32,32)
        self.crop_p = 0.500000
        self.dev_ids = None

        #argparser
        self.dataset_dir = dir_name
        self.epochs = epochs
        self.model = model
        self.device = device
        self.loss_fn = loss_fn

        self.batch_size = 64#512
        self.test_batch_size = 64#512
        self.num_workers = 8

        #model parameters
        self.a = 0.84
        self.k = 1
        self.in_channels = 1
        self.hidden_channels = 32
        self.lr = lr
        self.decay_lr = 0.9
        self.warmup_epoch = 999999999

        #logs
        self.plot_dataset = False
        self.plot_acts = False

        self.epoch_log = 1
        self.epoch_test_start = 0
        self.epoch_test = 1

        self.load = False
        self.load_epoch = 25

        self.save = True
        #self.epoch_save = 5

        #build directories
        if out_name == None:
            t = dtm.now().strftime("%y%m%d_%H%M%S")
            self.dir_output = "./denoising/output/%s"%t
        else:
            self.dir_output = "./denoising/output/%s"%out_name
        #self.dir_timings = self.dir_output + "/timings"
        self.dir_testing = self.dir_output + "/testing"
        self.dir_final_test = self.dir_output + "/final_test"
        self.dir_metrics = self.dir_output + "/metrics"
        self.dir_saved_models = self.dir_output + "/model_save"

        if not os.path.isdir(self.dir_output):
            os.mkdir(self.dir_output)
        if not os.path.isdir(self.dir_testing):
            os.mkdir(self.dir_testing)
        if not os.path.isdir(self.dir_final_test):
            os.mkdir(self.dir_final_test)
        if not os.path.isdir(self.dir_metrics):
            os.mkdir(self.dir_metrics)
        if not os.path.isdir(self.dir_saved_models):
            os.mkdir(self.dir_saved_models)
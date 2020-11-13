import os
from datetime import datetime as dtm

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.crop_size = (32,32)
        self.crop_p = 0.99 # signal to noise crops percentage

        #argparser
        self.val_batch_size = self.batch_size*4 if 'GCNN' in model else self.batch_size*8
        self.test_batch_size = self.batch_size*2 if 'GCNN' in model else self.batch_size*4
        self.num_workers = 8

        #model parameters
        self.a = 0.84
        self.k = 8
        self.in_channels = 1
        self.hidden_channels = 32
        
        #logs
        self.plot_dataset = False
        self.plot_acts = True

        self.epoch_log = 1
        self.epoch_test_start = 0
        self.epoch_test = 5

        self.t = 0.5

        self.load = False if (self.load_path is None) else True
        self.load_epoch = 100

        self.save = True
        #self.epoch_save = 5

    def build_directories(self):
        #build directories
        if self.out_name is None:
            t = dtm.now().strftime("%y%m%d_%H%M%S")
            self.dir_output = f"./denoising/output/{t}"
        else:
            self.dir_output = f"./denoising/output/{self.out_name}"

        def mkdir_fn(name):
            dirname = os.path.join(self.dir_output, name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            return dirname
        mkdir_fn("")
        self.dir_timings = mkdir_fn("timings")
        self.dir_testing = mkdir_fn("testing")
        self.dir_final_test = mkdir_fn("final_test")
        self.dir_metrics = mkdir_fn("metrics")
        self.dir_saved_models = mkdir_fn("model_save")


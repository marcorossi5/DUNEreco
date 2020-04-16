import os

class Args:
    def __init__(self, dir_name, epochs, model, device):
        #argparser
        self.dataset_dir = dir_name
        self.epochs = epochs
        self.model = model
        self.device = device

        self.batch_size = 256
        self.test_batch_size = 256
        self.num_workers = 8

        #model parameters
        self.k = 1
        self.in_channels = 1
        self.hidden_channels = 32
        self.lr = 1e-4
        self.decay_lr = 0.97

        #logs
        self.epoch_log = 10
        self.epoch_test_start = 1
        self.epoch_test = 10

        self.load = False
        self.load_epoch = 25

        self.save = True
        self.epoch_save = 25

        #build directories
        self.dir_output = "./denoising/output"
        self.dir_timings = self.dir_output + "/timings"
        self.dir_testing = self.dir_output + "/testing"
        self.dir_final_test = self.dir_output + "/final_test"
        self.dir_metrics = self.dir_output + "/metrics"
        self.dir_saved_models = self.dir_output + "/model_save"

        if not os.path.isdir(self.dir_output):
            os.mkdir(self.dir_output)
        if not os.path.isdir(self.dir_timings):
            os.mkdir(self.dir_timings)
        if not os.path.isdir(self.dir_testing):
            os.mkdir(self.dir_testing)
        if not os.path.isdir(self.dir_final_test):
            os.mkdir(self.dir_final_test)
        if not os.path.isdir(self.dir_metrics):
            os.mkdir(self.dir_metrics)
        if not os.path.isdir(self.dir_saved_models):
            os.mkdir(self.dir_saved_models)
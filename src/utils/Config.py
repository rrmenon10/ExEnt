import json
import os
import ast

from src.utils.util import make_exp_dir

class Config(object):
    def __init__(self, filename=None, kwargs=None, mkdir=True):

        # Model configs
        self.bench_type = "clues_syn"
        self.dataset = "synthetic"
        self.test_dataset = None
        self.task_num = None
        self.input_mode = 'ENTAIL'
        self.max_text_length = 64
        self.SYN_MAX = 144 # Maximum number of datasets in clues_syn

        self.pretrained_weight = "roberta-base"
        self.batch_size = 8
        self.eval_batch_size = 64
        self.num_batches = 1000
        self.eval_every = 1
        self.save_model = True

        self.eval_dev = True
        self.grad_accumulation_factor = 1

        self.exp_dir = None

        self.seed = 42
        self.lr = 1e-3
        self.optimizer_epsilon = 1e-8
        self.weight_decay = 0
        self.grad_clip_norm = 1
        self.lbl_max_text_length = 4
        
        if filename:
            self.__dict__.update(json.load(open(filename)))
        if kwargs:
            self.update_kwargs(kwargs)
        
        self.update_exp_config(mkdir)

    def update_kwargs(self, kwargs):
        for (k, v) in kwargs.items():
            try:
                v = ast.literal_eval(v)
            except ValueError:
                v = v
            except:
                v = v
            setattr(self, k, v)

    def update_exp_config(self, mkdir=True):
        '''
        Updates the config default values based on parameters passed in from config file
        '''
        self.base_dir = os.path.join("exp_out", self.bench_type)

        if mkdir:
            self.exp_dir = make_exp_dir(self.base_dir)

        if hasattr(self, 'exp_dir'):
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))

    def to_json(self):
        '''
        Converts parameter values in config to json
        :return: json
        '''
        return json.dumps(self.__dict__, indent=4, sort_keys=True)

    def save_config(self, filename):
        '''
        Saves the config
        '''
        with open(filename, 'w+') as fout:
            fout.write(self.to_json())
            fout.write('\n')

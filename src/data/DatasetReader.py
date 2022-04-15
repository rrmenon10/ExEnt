import numpy as np


from src.data.CLUESRealReader import RealReader
from src.data.CLUESSynReader import SyntheticReader

class DatasetReader(object):
    '''
    DatasetReader is responsible for reading dataset
    '''
    def __init__(self, config, tokenizer, dataset):
        '''
        :param config:
        :param tokenizer:
        :param dataset:
        '''
        self.config = config
        self.dataset = dataset

        if config.bench_type == 'clues_real':
            self.dataset_reader = RealReader(config, tokenizer, dataset)
        elif config.bench_type == 'clues_syn':
            assert len(self.dataset.lower().split("_"))>1, "Synthetic dataset configs should be provided with task number!"
            task_num = self.dataset.lower().split("_")[1]
            self.dataset_reader = SyntheticReader(config, tokenizer, task_num)
        else:
            raise ValueError("Invalid Dataset name")

    def read_dataset(self, split):
        '''
        Read dataset

        :param split:
        :param is_eval:
        :return:
        '''
        return np.asarray(self.dataset_reader.read_dataset(split))

    def prepare_batch(self, batch, type=""):
        '''
        Prepare batch of data for model

        :param batch:
        :param type: pattern to prepare batch with and which mode to use (ex: ENTAIL)
        :return:
        '''
        # Prepare for evaluation objective
        if "ENTAIL" in type:
            return self.dataset_reader.prepare_entail_batch(batch)
        else:
            raise ValueError("Invalid prepare_batch mode")
    
    @property
    def num_lbls(self):
        return self.dataset_reader.num_lbls
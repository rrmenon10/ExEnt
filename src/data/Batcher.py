from torch.utils import data
from src.data.Dataset import Dataset
from src.data.DatasetReader import DatasetReader
from src.utils.util import set_seeds


class Batcher(object):
    '''
    Batcher is responsible for returning batches of data
    '''
    def __init__(self, config, tokenizer, dataset):
        '''
        :param config:
        :param tokenizer:
        :param dataset:
        '''
        self.config = config
        self.dataset_reader = DatasetReader(config, tokenizer, dataset)
        set_seeds(self.config.seed)

        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None

    def get_dataset_reader(self):
        return self.dataset_reader

    @staticmethod
    def my_collate_fn(batch):

        dict_batch = {}
        dict_batch["input"] = {}
        dict_batch["output"] = {}

        for datapoint in batch:
            for (k, v) in datapoint["input"].items():
                if k in dict_batch["input"]:
                    dict_batch["input"][k].append(v)
                else:
                    dict_batch["input"][k] = [v]

            for (k, v) in datapoint["output"].items():
                if k in dict_batch["output"]:
                    dict_batch["output"][k].append(v)
                else:
                    dict_batch["output"][k] = [v]

        return dict_batch

    def _init_train(self):
        '''
        Initialize loader for train data
        '''
        train_data = self.dataset_reader.read_dataset("train")
        self.train_loader = data.DataLoader(Dataset(train_data), batch_size=self.config.batch_size, shuffle=True, collate_fn=self.my_collate_fn)


    def _init_dev(self):
        '''
        Initialize loader for dev data
        '''
        dev_data = self.dataset_reader.read_dataset("dev")
        self.dev_loader = data.DataLoader(Dataset(dev_data), batch_size=self.config.eval_batch_size, shuffle=False, collate_fn=self.my_collate_fn)

    def _init_test(self):
        '''
        Initialize loader for test data
        '''
        test_data = self.dataset_reader.read_dataset("test")
        self.test_loader = data.DataLoader(Dataset(test_data), batch_size=self.config.eval_batch_size, shuffle=False, collate_fn=self.my_collate_fn)
    
    def get_train_batch(self):
        '''
        Yield train batches

        :return:
        '''
        if self.train_loader is None:
            self._init_train()

        while True:
            for x in self.train_loader:
                yield x

    def get_dev_batch(self):
        '''
        Yield dev batches

        :return:
        '''
        if self.dev_loader is None:
            self._init_dev()

        for x in self.dev_loader:
            yield x


    def get_test_batch(self):
        '''
        Yield test batches

        :return:
        '''
        if self.test_loader is None:
            self._init_test()

        for x in self.test_loader:
            yield x
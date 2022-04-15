import os
import torch
import argparse
from transformers import AutoTokenizer
import logging

from src.eval.eval_model import multi_task_test_eval
from src.data.Batcher import Batcher
from src.ExEnt import ExEnt

from src.utils.Config import Config
from src.utils.util import set_global_logging_level
from src.utils.util import device

set_global_logging_level(logging.ERROR)


def test(config):

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)

    dict_test_batcher = {}
    dict_test_dataset_reader = {}

    max_datasets = config.SYN_MAX

    if config.test_dataset:
        if "<" in config.test_dataset:
            max_num = int(config.test_dataset.split("_<")[-1])
            config.test_dataset = [f"synthetic_{i+1}" for i in range(max_num)]
        elif ">" in config.test_dataset:
            start_num = int(config.test_dataset.split("_>")[-1])
            config.test_dataset = [f"synthetic_{i+1}" for i in range(start_num, max_datasets)]

    assert config.test_dataset is not None, "Need to provide some test datasets"
    for dataset in config.test_dataset:
        batcher = Batcher(config, tokenizer, dataset)
        dict_test_batcher[dataset] = batcher

        dataset_reader = batcher.get_dataset_reader()
        dict_test_dataset_reader[dataset] = dataset_reader

    model = ExEnt(config, tokenizer, dict_test_dataset_reader).to(device)
    model.load_state_dict(torch.load(os.path.join(config.exp_dir, "best_model.pt")))
    multi_task_test_eval(config, model, dict_test_batcher)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_dir', required=True)
    args = parser.parse_args()
    
    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, mkdir=False)
    
    test(config)
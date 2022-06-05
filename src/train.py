import os
import torch
import argparse
import logging

from transformers import AutoTokenizer

from src.eval.eval_model import multi_task_dev_eval
from src.data.Batcher import Batcher

from src.utils.Config import Config
from src.utils.get_optimizer import get_optimizer
from src.utils.get_model import get_model
from src.utils.util import get_avg_dict_val_store, update_dict_val_store, ParseKwargs
from src.utils.util import set_global_logging_level
from src.utils.util import device

set_global_logging_level(logging.ERROR)


def train(config):
    '''
    Trains the model
    :param config:
    :return:
    '''
    max_datasets = config.SYN_MAX

    if "<" in config.dataset:
        max_num = int(config.dataset.split("_<")[-1])
        config.dataset = [f"synthetic_{i+1}" for i in range(max_num)]
    elif ">" in config.dataset:
        start_num = int(config.dataset.split("_>")[-1])
        config.dataset = [f"synthetic_{i+1}" for i in range(start_num, max_datasets)]

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_weight)

    dict_batcher = {}
    dict_train_iter = {}
    dict_dataset_reader = {}

    # Instantiate dataset readers and batchers for each dataset
    for dataset in config.dataset:
        batcher = Batcher(config, tokenizer, dataset)
        dict_batcher[dataset] = batcher

        train_iter = batcher.get_train_batch()
        dict_train_iter[dataset] = train_iter

        dataset_reader = batcher.get_dataset_reader()
        dict_dataset_reader[dataset] = dataset_reader

    # Initialize model and optimizer
    model = get_model(config, tokenizer, dict_dataset_reader)
    optimizer = get_optimizer(model, config)
    
    best_dev_acc = 0
    dict_val_store = None

    tot_num_batches = config.num_batches * config.grad_accumulation_factor

    for i in range(tot_num_batches):
        # Get true batch_idx
        batch_idx = (i // config.grad_accumulation_factor) + 1

        task_dataset = config.dataset[batch_idx % len(config.dataset)]

        model.train()
        train_batch = next(dict_train_iter[task_dataset])
        
        loss, dict_val_update = model(train_batch, task_dataset)
        loss = loss / config.grad_accumulation_factor
        loss.backward()

        if (i+1) % config.grad_accumulation_factor == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()


        dict_val_store = update_dict_val_store(dict_val_store, dict_val_update, config.grad_accumulation_factor)
        print("Finished %d batches" % batch_idx, end='\r')

        if batch_idx % config.eval_every == 0 and i % config.grad_accumulation_factor == 0:
            dict_avg_val = get_avg_dict_val_store(dict_val_store, config.eval_every)
            dict_val_store = None

            # Compute dev accuracy
            rng_state = torch.get_rng_state()
            dev_acc_iid = multi_task_dev_eval(config, model, dict_batcher, batch_idx, dict_avg_val)
            torch.set_rng_state(rng_state)

            print("Global Step: %d Acc_iid: %.3f" % (batch_idx, dev_acc_iid) + '\n')

            if config.save_model:
                if dev_acc_iid > best_dev_acc:
                    best_dev_acc = dev_acc_iid
                    torch.save(model.state_dict(), os.path.join(config.exp_dir, "best_model.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-m', '--mkdir', action='store_false')
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=args.mkdir)

    train(config)

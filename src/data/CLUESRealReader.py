import os
import json
import torch

from src.utils.util import device
from src.data.tokenize_txt import tokenize_txt


class RealReader(object):
    '''
    RealReader reads real-world dataset
    '''

    def __init__(self, config, tokenizer, dataset):
        
        self.config = config
        self.dataset = dataset
        self.dataset_class, self.dataset_name = self.dataset.split("/")
        self.tokenizer = tokenizer

        detail_file = os.path.join('data', 'clues_real_utils', 'real_task_details.json')
        with open(detail_file, 'r') as f:
            task_details = json.load(f)[self.dataset_name]
        
        explanations_file = os.path.join('data', 'clues_real_utils', 'real_lbl_indicator_map.json')
        with open(explanations_file, "r") as f:
            self.explanations = list(json.load(f)[self.dataset_name].keys())
        
        lbl_indicator_file = os.path.join('data', 'clues_real_utils', 'real_lbl_indicator_map.json')
        with open(lbl_indicator_file, "r") as f:
            self._real_lbl_indicator_map = json.load(f)[self.dataset_name]
        
        exp_entail_indicator_file = os.path.join('data', 'clues_real_utils', 'real_exp_entail_indicator.json')
        with open(exp_entail_indicator_file, "r") as f:
            self._real_exp_entail_indicator_map = json.load(f)[self.dataset_name]
        
        self.columns = task_details['columns']
        self.task_lbls = task_details['task_lbls']
        
        self.dict_lbl_2_idx = {lbl:idx for idx, lbl in enumerate(self.task_lbls)}
        self.dict_idx_2_lbl = {idx:lbl for idx, lbl in enumerate(self.task_lbls)}
        self.num_lbls = len(self.task_lbls)

    def _get_file(self, split):
        '''
        Get filename of split
        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("data", self.config.bench_type, self.dataset_class, self.dataset_name, f"train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", self.config.bench_type, self.dataset_class, self.dataset_name, f"dev.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", self.config.bench_type, self.dataset_class, self.dataset_name, f"test.jsonl")
        return file

    def read_dataset(self, split):
        '''
        Read the dataset
        :param split: partition of the dataset
        :return:
        '''
        split_file = self._get_file(split)
        
        data = []

        with open(split_file, 'r') as f_in:
            for idx, line in enumerate(f_in.readlines()):
                json_string = json.loads(line)

                dict_input = {}
                for col in json_string.keys():
                    if col in self.columns: dict_input[col] = json_string[col]
                dict_input["explanations"] = self.explanations
                dict_input["idx"] = idx

                dict_output = {}
                dict_output["lbl"] = self.dict_lbl_2_idx[str(json_string["lbl"])]
                dict_output["target_key"] = json_string["target_key"]

                dict_input_output = {"input": dict_input, "output": dict_output}
                data.append(dict_input_output)

        return data
    
    def prepare_entail_batch(self, batch):
        # Prepares batch for ExEnt model
        explanations = batch["input"]["explanations"]
        list_lbls = batch["output"]["lbl"]

        list_input_ids = []
        list_entail_flag = []
        list_exp_mask = []
        list_lbl_indicator = []

        max_num_exp = max([len(x) for x in explanations]) # keeping this for safety

        entail_class_map = {"entail":0, "contradict":1, "neutral": 2}

        for b_idx in range(len(list_lbls)):
            # First, table operations
            table = self.get_features(batch, b_idx)
            table_txt_features = f" {self.tokenizer.sep_token} ".join(table)
            list_exp_input_ids = []
            list_entail_flag_per_exp = []
            list_lbl_indicator_per_exp = []
            list_exp_mask_per_row = []

            # Next, we create each input-explanation pair for inputs to the ExEnt model
            for explanation in explanations[b_idx]:
                table_txt = explanation + f" {self.tokenizer.sep_token} " + table_txt_features
                
                # Get tokenized inputs
                input_ids = tokenize_txt(self.tokenizer, self.config.max_text_length, table_txt)
                list_exp_input_ids.append(input_ids)

                list_entail_flag_per_exp.append(entail_class_map[self._real_exp_entail_indicator_map[explanation]])
                list_exp_mask_per_row.append(1)
                list_lbl_indicator_per_exp.append(self.dict_lbl_2_idx[self._real_lbl_indicator_map[explanation]])
            
            # padding
            num_exp = len(explanations[b_idx])
            zero_mask = [0] * (max_num_exp - num_exp)
            list_exp_mask_per_row.extend(zero_mask)
            list_entail_flag_per_exp.extend(zero_mask)
            list_lbl_indicator_per_exp.extend(zero_mask)
            list_exp_input_ids.extend([[0]*self.config.max_text_length] * (max_num_exp - num_exp))

            list_input_ids.append(list_exp_input_ids)               # bs x max_text_length (inputs to the ExEnt model)
            list_entail_flag.append(list_entail_flag_per_exp)       # bs x max_num_exp     (info of whether exp. is of entailment/contradiction type)
            list_exp_mask.append(list_exp_mask_per_row)             # bs x max_num_exp     (mask for applicable explanations)
            list_lbl_indicator.append(list_lbl_indicator_per_exp)   # bs x max_num_exp     (info about class label mentioned by explanation)

        return torch.tensor(list_input_ids).to(device),  \
                torch.tensor(list_entail_flag).to(device), \
                torch.tensor(list_exp_mask).to(device), \
                torch.tensor(list_lbl_indicator).to(device), \
                torch.tensor(batch["output"]["lbl"]).to(device)
 
    def get_features(self, batch, b_idx):
        return [f"{col} | {batch['input'][col][b_idx]}" for col in self.columns if col in batch['input'].keys()]

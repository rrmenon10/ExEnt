import os
import json
import torch
import pandas as pd

from src.utils.util import device
from src.data.tokenize_txt import tokenize_txt


class SyntheticReader(object):
    '''
    SyntheticReader reads synthetic dataset
    '''

    def __init__(self, config, tokenizer, task_num):
        self.config = config
        self.tokenizer = tokenizer
        self.task_num = task_num

        description_file = os.path.join("data", self.config.bench_type, f"task_{self.task_num}", "description.jsonl")
        with open(description_file, 'r') as f_in:
            description = json.loads(f_in.readline())

        self.task_lbls = description['labels']
        self.columns = description['columns']

        self.dict_lbl_2_idx = {lbl:idx for idx, lbl in enumerate(self.task_lbls)}
        self.dict_idx_2_lbl = {idx:lbl for idx, lbl in enumerate(self.task_lbls)}
        self.explanations = self.get_explanations()
        self.num_lbls = len(self.task_lbls)
        
        
    def _get_file(self, split):
        '''
        Get filename of split
        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join("data", self.config.bench_type, f"task_{self.task_num}", "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join("data", self.config.bench_type, f"task_{self.task_num}", "valid.jsonl")
        elif split.lower() == "test":
            file = os.path.join("data", self.config.bench_type, f"task_{self.task_num}", "test.jsonl")
        return file

    def get_explanations(self):
        explanations_file = os.path.join("data", self.config.bench_type, f"task_{self.task_num}", "explanations.csv")
        explanations = pd.read_csv(explanations_file, header=None)
        explanations = [val for vals in explanations.values.tolist() for val in vals]
        return explanations

    def read_dataset(self, split):
        '''
        Read the dataset
        :param split: partition of the dataset
        :return:
        '''
        file = self._get_file(split)

        data = []
        
        attribute_map = json.load(open(os.path.join("data", "clues_syn_attributes.json")))
        all_target_names = [list(attr['targets'].keys())[0] for attr in attribute_map]

        with open(file, 'r') as f_in:
            for idx, line in enumerate(f_in.readlines()):
                json_string = json.loads(line)

                dict_input = {}
                for col in self.columns.keys():
                    dict_input[col] = json_string[col]
                dict_input["idx"] = idx
                dict_input["explanations"] = self.explanations
                dict_input["task_num"] = self.task_num
                dict_output = {}
                target_key = [key for key in all_target_names if key in json_string][0]
                if target_key in json_string:
                    dict_output["lbl"] = self.dict_lbl_2_idx[json_string[target_key]]
                else:
                    dict_output["lbl"] = -1
                dict_output["target_key"] = target_key

                dict_input_output = {"input": dict_input,
                                     "output": dict_output,
                                    }
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

        max_num_exp = max([len(x) for x in explanations])

        for b_idx in range(len(list_lbls)):
            
            # First, table operations
            table = self.get_features(batch, b_idx)
            table_txt_features = f" {self.tokenizer.sep_token} ".join(table)
            list_exp_input_ids = []
            list_entail_flag_per_exp = []
            list_lbl_indicator_per_exp = []
            list_exp_mask_per_row = []
            for explanation in explanations[b_idx]:
                
                table_txt = explanation + f" {self.tokenizer.sep_token} " + table_txt_features

                # Get tokenized inputs
                input_ids = tokenize_txt(self.tokenizer, self.config.max_text_length, table_txt)
                list_exp_input_ids.append(input_ids)

                # check if explanation is of entailment/contradiction type
                if "then not not" in explanation:
                    list_entail_flag_per_exp.append(0) # 0 denotes entailment logit index
                elif "then not" in explanation:
                    list_entail_flag_per_exp.append(1) # 1 denotes contradiction logit index
                else:
                    list_entail_flag_per_exp.append(0) # 0 denotes entailment logit index
                
                # explanation applicable here
                list_exp_mask_per_row.append(1)

                ## The rest of the operations below effectively reads information about the class label from explantion.
                # get part after "then" and check which label is present
                if "it is" in explanation:
                    part_before_it_is = explanation.split("it is")[0]
                    split_idx = 2
                    part_after_it_is = " ".join(explanation.split("it is")[1].split(" ")[split_idx:])

                    explanation_modified = part_before_it_is + " " + part_after_it_is
                    explanation_modified = " ".join(explanation_modified.split())
                else:
                    explanation_modified = explanation
                
                if "then not not" in explanation_modified:
                    part_after_then = explanation_modified.split("then not not ")[1]
                elif "then not" in explanation_modified and any("not" in label for label in self.task_lbls):
                    part_after_then = explanation_modified.split("then ")[1]
                elif "then not" in explanation_modified and not any("not" in label for label in self.task_lbls):
                    part_after_then = explanation_modified.split("then not ")[1]
                else:
                    part_after_then = explanation_modified.split("then ")[1]
                
                part_after_then = part_after_then.strip().strip(".")
                assert (part_after_then in list(self.dict_lbl_2_idx.keys())) == True
                
                # get label indicators
                for _lbl, _lbl_idx in self.dict_lbl_2_idx.items():
                    if _lbl == part_after_then:
                        list_lbl_indicator_per_exp.append(_lbl_idx)
                        break

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
            list_lbl_indicator.append(list_lbl_indicator_per_exp)   # bs x max_num_exp     (info about class label mentioned by exp.)

        return torch.tensor(list_input_ids).to(device),  \
                torch.tensor(list_entail_flag).to(device), \
                torch.tensor(list_exp_mask).to(device), \
                torch.tensor(list_lbl_indicator).to(device), \
                torch.tensor(batch["output"]["lbl"]).to(device)

    def get_features(self, batch, b_idx):
        return [f"{col} | {batch['input'][col][b_idx]}" for col in self.columns.keys()]
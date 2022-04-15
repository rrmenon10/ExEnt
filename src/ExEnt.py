import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification


NUM_ENTAIL_CLASSES = 2

class ExEnt(nn.Module):
    def __init__(self,config, tokenizer, dataset_reader):
        super(ExEnt, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

        if "roberta" in self.config.pretrained_weight:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.pretrained_weight)
        else:
            raise ValueError

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch, dataset=None):

        assert dataset is not None, 'Need to pass dataset name as input'
        dataset_reader = self.dataset_reader[dataset]
        input_mode = self.config.input_mode + "_" * (len(self.config.input_mode)>0)
        processed_inputs = dataset_reader.prepare_batch(batch, type=f"{input_mode}TRAIN")
        input_ids, entail_flag, exp_mask, exp_lbl_indicator, lbls = processed_inputs
        # entail_flag       bs x max_num_explanations : gives the number of explanations 
        #                                           that need us to look at the
        #                                           entailment(0)/contradiction(1) score
        # exp_mask          bs x max_num_explanations: gives the number of explanations 
        #                                          applicable to each example. Useful with shuffle
        # exp_lbl_indicator bs x max_num_explanations: gives the corresponding label 
        #                                          mentioned by each explanation.
        # lbls              bs: gives the actual labels of the examples.

        bs, num_exp, seq_length = input_ids.size()
        num_lbls = dataset_reader.num_lbls
        input_ids = input_ids.view(bs*num_exp, seq_length)
        exp_mask = exp_mask.unsqueeze(-1)
        
        # Get attention masks for each input
        attention_masks = (input_ids != self.tokenizer.pad_token_id).long()

        ## Forward pass and compute entailment logits
        outputs = self.model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        logits = outputs.logits # (bs * max_num_explanations) * num_classes
        logits = logits.view(bs, num_exp, -1)
        
        ## Convert entailment logits to class logits
        # Next, we will construct the true and false flags; 
        # true_flag corresponds to reading the required(entailment/contradiction) 
        # scores for each explanation.
        true_flag = F.one_hot(entail_flag, num_classes=NUM_ENTAIL_CLASSES)
        false_flag = 1 - true_flag
        true_logits  = torch.sum(logits[...,:NUM_ENTAIL_CLASSES] * true_flag * exp_mask, dim = -1, keepdim=True)
        false_logits = torch.sum(logits[...,:NUM_ENTAIL_CLASSES] * false_flag * exp_mask, dim = -1, keepdim=True) / (num_lbls - 1)
        
        exp_lbls_one_hot = F.one_hot(exp_lbl_indicator.view(-1), num_classes=num_lbls).view(bs, num_exp, -1)
        exp_lbls_one_hot = exp_lbls_one_hot * exp_mask
        
        neutral_logits = (logits[...,-1:] * exp_mask) / num_lbls
        
        ce_logits = true_logits.repeat(1, 1, num_lbls) * exp_lbls_one_hot + \
                    false_logits.repeat(1, 1, num_lbls) * (1. - exp_lbls_one_hot) + \
                    neutral_logits.repeat(1, 1, num_lbls)
        
        ## Aggregate all explanation information
        ce_logits = ce_logits.mean(dim=1)            
        loss = self.ce_loss(ce_logits, lbls.long())
        
        dict_val = {"loss": loss}
        return loss, dict_val

    def predict(self, batch, dataset=None):

        assert dataset is not None, 'Need to pass dataset name as input'
        dataset_reader = self.dataset_reader[dataset]
        input_mode = self.config.input_mode + "_" * (len(self.config.input_mode)>0)
        processed_inputs = dataset_reader.prepare_batch(batch, type=f"{input_mode}EVAL")
        input_ids, entail_flag, exp_mask, exp_lbl_indicator, lbls = processed_inputs

        bs, num_exp, seq_length = input_ids.size()
        num_lbls = dataset_reader.num_lbls
        input_ids = input_ids.view(bs*num_exp, seq_length)
        exp_mask = exp_mask.unsqueeze(-1)

        # Get attention masks for each input
        attention_masks = (input_ids != self.tokenizer.pad_token_id).long()

        ## Forward pass and compute entailment logits
        outputs = self.model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        logits = outputs.logits # (bs * max_num_explanations) * num_classes
        logits = logits.view(bs, num_exp, -1)
        
        ## Convert entailment logits to class logits
        # Next, we will construct the true and false flags; 
        # true_flag corresponds to reading the required(entailment/contradiction) 
        # scores for each explanation.
        true_flag = F.one_hot(entail_flag, num_classes=NUM_ENTAIL_CLASSES)
        false_flag = 1 - true_flag
        true_logits  = torch.sum(logits[...,:NUM_ENTAIL_CLASSES] * true_flag * exp_mask, dim = -1, keepdim=True)
        false_logits = torch.sum(logits[...,:NUM_ENTAIL_CLASSES] * false_flag * exp_mask, dim = -1, keepdim=True) / (num_lbls - 1)
        
        exp_lbls_one_hot = F.one_hot(exp_lbl_indicator.view(-1), num_classes=num_lbls).view(bs, num_exp, -1)
        exp_lbls_one_hot = exp_lbls_one_hot * exp_mask
        
        neutral_logits = (logits[...,-1:] * exp_mask) / num_lbls
        
        ce_logits = true_logits.repeat(1, 1, num_lbls) * exp_lbls_one_hot + \
                    false_logits.repeat(1, 1, num_lbls) * (1. - exp_lbls_one_hot) + \
                    neutral_logits.repeat(1, 1, num_lbls)

        ## Aggregate all explanation information
        ce_logits = ce_logits.mean(dim=1)
        pred_lbls = ce_logits.argmax(dim=-1).cpu().numpy()
        true_lbls = lbls.cpu().numpy()

        return pred_lbls, true_lbls, ce_logits
import torch
import torch.nn as nn

from transformers import RobertaConfig, RobertaModel

from src.utils.util import l2_normalize

class ZeroShotClassifier(nn.Module):

    def __init__(self, config, tokenizer, dataset_reader):
        '''
        RoBERTa Zero-shot Classifier model
        '''
        super(ZeroShotClassifier, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

        if "roberta" in self.config.pretrained_weight:
            config = RobertaConfig.from_pretrained(self.config.pretrained_weight)
            self.model = RobertaModel.from_pretrained(self.config.pretrained_weight)
        else:
            raise ValueError

        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, batch, dataset=None):

        assert dataset is not None, 'Need to pass dataset name as input'
        dataset_reader = self.dataset_reader[dataset]
        input_mode = self.config.input_mode + "_" * (len(self.config.input_mode)>0)
        input_ids, lbl_ids, lbls = dataset_reader.prepare_batch(batch, type=f"{input_mode}REP")

        # Get attention masks for each input-output
        input_attention_masks = (input_ids != self.tokenizer.pad_token_id).long()
        lbl_attention_masks = (lbl_ids != self.tokenizer.pad_token_id).long()

        # Get representation of inputs
        context_rep = self.model(input_ids, attention_mask=input_attention_masks).pooler_output
        
        # Get representation of labels
        with torch.no_grad():
            lbl_rep = self.model(lbl_ids, attention_mask=lbl_attention_masks).pooler_output
        
        # Compute input-label alignment with dot product
        logits = torch.matmul(l2_normalize(context_rep), l2_normalize(lbl_rep).T)
        
        loss = self.ce_loss(logits, lbls.long())
        
        dict_val = {"loss": loss}
        return loss, dict_val
    
    def predict(self, batch, dataset=None):

        assert dataset is not None, 'Need to pass dataset name as input'
        dataset_reader = self.dataset_reader[dataset]
        input_mode = self.config.input_mode + "_" * (len(self.config.input_mode)>0)
        input_ids, lbl_ids, lbls = dataset_reader.prepare_batch(batch, type=f"{input_mode}REP")

        # Get attention masks for each input-output
        input_attention_masks = (input_ids != self.tokenizer.pad_token_id).long()
        lbl_attention_masks = (lbl_ids != self.tokenizer.pad_token_id).long()

        # Get representation of inputs
        context_rep = self.model(input_ids, attention_mask=input_attention_masks).pooler_output
        
        # Get representation of labels
        with torch.no_grad():
            lbl_rep = self.model(lbl_ids, attention_mask=lbl_attention_masks).pooler_output
        
        # Compute input-label alignment with dot product
        logits = torch.matmul(l2_normalize(context_rep), l2_normalize(lbl_rep).T)
        
        pred_lbls = logits.argmax(dim=-1).cpu().numpy()
        true_lbls = lbls.cpu().numpy()
        return pred_lbls, true_lbls, logits
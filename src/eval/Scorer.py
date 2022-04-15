import torch
import re
import os
import json
import glob
import click
import string
import tqdm
import pickle
import numpy as np
import collections

from collections import defaultdict, Counter

from src.utils.Config import Config

class Scorer(object):

    def __init__(self, config, dataset=None):
        self.config = config

        # Metrics to compute
        self.compute_acc = True
        dataset = self.config.dataset if dataset is None else dataset
        self.dict_idx2logits_lbl = {}

    def _compute_acc(self):
        '''
        :return:
        '''
        acc_cor_cnt = 0
        acc_ttl_cnt = 0

        for (_, pred_true_lbl) in self.dict_idx2logits_lbl.items():
            pred_lbl = pred_true_lbl[0][0]
            true_lbl = pred_true_lbl[0][1]

            acc_ttl_cnt += 1
            if pred_lbl == true_lbl:
                acc_cor_cnt += 1

        round_tot_acc = float(round(acc_cor_cnt / acc_ttl_cnt, 3))
        return round_tot_acc

    def add_batch(self, list_idx, list_pred_lbl, list_true_lbl, lbl_logits):
        '''
        Keeps track of the accuracy of current batch
        :param logits:
        :param true_lbl:
        :return:
        '''
        for idx, pred_lbl, true_lbl, logit in zip(list_idx, list_pred_lbl, list_true_lbl, lbl_logits):
            if idx in self.dict_idx2logits_lbl:
                self.dict_idx2logits_lbl[idx].append((pred_lbl, true_lbl, logit))
            else:
                self.dict_idx2logits_lbl[idx] = [(pred_lbl, true_lbl, logit)]

    def get_score(self, split):
        '''
        Gets the accuracy
        :return: rounded accuracy to 3 decimal places
        '''

        dict_scores = {}
        score_eval = 0

        round_tot_acc = self._compute_acc()
        type = "%s_acc" % split
        dict_scores[type] = round_tot_acc
        score_eval = round_tot_acc
        
        return score_eval, dict_scores
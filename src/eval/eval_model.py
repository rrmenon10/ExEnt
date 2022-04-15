import json
import torch

from src.eval.Scorer import Scorer


def multi_task_dev_eval(config, model, dict_batcher, num_batches, dict_avg_val=None):
    '''
    Evaluates the accuracy on the dev partition
    :param config:
    :param model:
    :param num_batches:
    :param loss:
    :return: currrent dev accuracy or average of previous split dev accuracy
    '''

    dict_eval = {}
    score_eval = []
    dict_eval["num_batches"] = num_batches

    if dict_avg_val is not None:
        dict_eval.update(dict_avg_val)

    model.eval()

    # Get dev Score
    list_dataset = list(dict_batcher.keys())

    for dataset in list_dataset:
        batcher = dict_batcher[dataset]
        dataset_eval = {}

        # Get dev Score
        if config.eval_dev:
            dev_scorer = Scorer(config, dataset)
            with torch.no_grad():
                for batch in batcher.get_dev_batch():
                    pred_lbl, true_lbl, lbl_logits = model.predict(batch, dataset)
                    list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"]["idx"].cpu().numpy().tolist()
                    dev_scorer.add_batch(list_idx, pred_lbl, true_lbl, lbl_logits.cpu().numpy())
            dataset_score_eval, dev_scores = dev_scorer.get_score("dev")
            dataset_eval.update(dev_scores)
        else:
            dataset_score_eval = 0

        score_eval.append(dataset_score_eval)

        dict_eval[dataset] = dataset_eval

    with open(config.dev_score_file, 'a+') as f_out:
        f_out.write(json.dumps(dict_eval))
        f_out.write('\n')

    return sum(score_eval) / len(score_eval)

def multi_task_test_eval(config, model, dict_batcher, dict_avg_val=None):
    '''
    Evaluates the accuracy on the test partition
    :param config:
    :param model:
    '''

    dict_eval = {}
    score_eval = []

    if dict_avg_val is not None:
        dict_eval.update(dict_avg_val)

    model.eval()

    # Get test Score
    list_dataset = list(dict_batcher.keys())

    for dataset in list_dataset:
        batcher = dict_batcher[dataset]
        dataset_eval = {}

        # Get test Score
        test_scorer = Scorer(config, dataset)
        with torch.no_grad():
            for batch in batcher.get_test_batch():
                pred_lbl, true_lbl, lbl_logits = model.predict(batch, dataset)
                list_idx = batch["input"]["idx"] if isinstance(batch["input"]["idx"], list) else batch["input"]["idx"].cpu().numpy().tolist()
                test_scorer.add_batch(list_idx, pred_lbl, true_lbl, lbl_logits.cpu().numpy())
        dataset_score_eval, test_scores = test_scorer.get_score("test")
        dataset_eval.update(test_scores)

        score_eval.append(dataset_score_eval)

        dict_eval[dataset] = dataset_eval

    with open(config.test_score_file, 'w') as f_out:
        f_out.write(json.dumps(dict_eval))
        f_out.write('\n')
from transformers import AdamW


def get_optimizer(model, config):
    '''
    Construct optimizer based on config
    :param config:
    :param model:
    :return:
    '''

    # Ignore decay for larger model
    if "roberta" in config.pretrained_weight:
        no_decay = ['bias', 'LayerNorm.weight']
    else:
        no_decay = []

    # Ignore decay for certain parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay,
         'lr': config.lr},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': config.lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=config.optimizer_epsilon)

    return optimizer
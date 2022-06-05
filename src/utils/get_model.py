from src.models.ExEnt import ExEnt
from src.models.ZSC import ZeroShotClassifier

from src.utils.util import device

def get_model(config, tokenizer, dataset_reader):
    '''
    Creates the model based on config
    :param config: configuration for creating model
    :return: model
    '''

    if config.model == "ExEnt":
        model = ExEnt(config, tokenizer, dataset_reader).to(device)
    elif config.model == "ZeroShotClassifier":
        model = ZeroShotClassifier(config, tokenizer, dataset_reader).to(device)
    else:
        raise ValueError(f"Invalid Model {config.model}")
    return model
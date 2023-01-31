from functools import partial
from models.pretraining.graph_infomax import random_sample_corruption, DeepGraphInfomaxWrapper as DGI

# TODO get the training set correctly
TRAINING_SET = None

CORRUPTION_FUNC = partial(
    random_sample_corruption,
    TRAINING_SET
)



def main():
    model = DGI




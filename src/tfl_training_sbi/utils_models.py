"""Utility function to support the handling of models. """


import pickle as pl
import sbi.inference


def save_posterior_obj(path: str, model: sbi.inference.NeuralInference) -> None:
    """Saves a sbi Neural Posterior as pickle. 

    Args:
        path (str): Path incl. file name.
        model (sbi.inference.NeuralInference): Neural Inference obj. from sbi.
    """
    with open(path, "wb") as file:
        pl.dump(model, file)

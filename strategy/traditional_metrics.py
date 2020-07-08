import numpy as np
import constants
import os

# Mostly based off of https://arxiv.org/pdf/1301.5943.pdf
def compute_aggression(strategy):
  n_calls = np.sum(strategy[..., constants.CALL])
  n_raises = np.sum(strategy[..., constants.RAISE + 1:])
  if n_calls > 0:
    return n_raises / n_calls
  else:
    return np.inf

def compute_tightness(strategy):
  n_folds = np.sum(strategy[..., constants.FOLD])
  n_calls = np.sum(strategy[..., constants.CALL])
  n_raises = np.sum(strategy[..., constants.RAISE + 1:])
  return n_folds / (n_calls + n_raises + n_folds)

def tightness_border():
  # https://arxiv.org/pdf/1301.5943.pdf page 73
  return 0.72
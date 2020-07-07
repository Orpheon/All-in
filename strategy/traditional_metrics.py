import numpy as np
import constants
import os

# Mostly based off of https://arxiv.org/pdf/1301.5943.pdf
def compute_playstyle(strategy):
  n_folds = np.sum(strategy[..., constants.FOLD])
  n_calls = np.sum(strategy[..., constants.CALL])
  n_checks = np.sum(strategy[..., constants.CALL + 1])
  n_raises = np.sum(strategy[..., constants.RAISE + 1:])

  if n_calls > 0:
    aggression_factor = n_raises / n_calls
  else:
    aggression_factor = np.inf
  tightness = n_folds / (n_calls + n_raises + n_folds)

  return aggression_factor, tightness

def tightness_border():
  # https://arxiv.org/pdf/1301.5943.pdf page 73
  return 0.72
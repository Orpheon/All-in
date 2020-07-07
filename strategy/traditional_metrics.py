import numpy as np
import constants
import os

# Mostly based off of https://arxiv.org/pdf/1301.5943.pdf
def compute_playstyle(strategy):
  n_folds = np.count_nonzero(strategy[..., constants.FOLD])
  n_calls = np.count_nonzero(strategy[..., constants.CALL])
  n_checks = np.count_nonzero(strategy[..., constants.CALL + 1])
  n_raises = np.count_nonzero(strategy[..., constants.RAISE + 1:])
  # print("folds", n_folds)
  # print("calls", n_calls)
  # print("checks", n_checks)
  # print("raises", n_raises)

  aggression_factor = n_raises / (n_calls + 1e-9)
  tightness = n_folds / (n_calls + n_raises)

  return aggression_factor, tightness

def tightness_border():
  # https://arxiv.org/pdf/1301.5943.pdf page 73
  return 0.72
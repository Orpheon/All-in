import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from strategy.traditional_metrics import compute_playstyle, tightness_border

sns.set_style('whitegrid')

strategy_path = os.path.join("strategy", "strat_vectors")
strategy_files = os.listdir(strategy_path)

data = pd.DataFrame()

for file in strategy_files:
  strategy = np.load(os.path.join(strategy_path, file))
  agent_id = file.replace("_strategy.npy", "")
  aggression, tightness = compute_playstyle(strategy)
  data = data.append({
    'agent': agent_id,
    'raw_tightness': tightness,
    'raw_aggression': aggression,
    'normed_tightness': tightness - tightness_border(),
    'log_aggression': np.log(aggression)
  }, ignore_index=True)

print(data)
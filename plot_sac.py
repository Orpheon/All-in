import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

data = pd.read_csv(os.path.join('sac1', 'logs', 'progress.csv'))
try:
  shutil.rmtree(os.path.join('sac1', 'logs', 'images'))
except:
  pass
os.makedirs(os.path.join('sac1', 'logs', 'images'), exist_ok=True)
for col in data:
  data[col].plot()
  plt.savefig(os.path.join('sac1', 'logs', 'images', col))
  plt.clf()
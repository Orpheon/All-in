import json
import numpy as np

with open("preflop_monte_carlo_winrates.json", "r") as f:
  preflop_table = json.load(f)

table = sorted(preflop_table, key=lambda x: x[1])
cards_only = [x[0] for x in table]

np.save("preflop_ranks.npy", np.array(cards_only))
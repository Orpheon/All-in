import treys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

# The purpose of this script is to generate a table with an approximation, for each hand value, how many hands (in %)
# that hand value beats, since they are not equally distributed
# It uses monte-carlo. Presumably this table exists as true calculation, but I can't find one

N_SCORES = 7642
PRINT_FREQUENCY = 100000
N_BINS = 20
N_LOOPS = 15

def evaluate_table(n_cards):
  stats = np.zeros(N_SCORES)

  deck = treys.Deck()
  evaluator = treys.Evaluator()
  for k in range(N_LOOPS):
    for l in range(PRINT_FREQUENCY):
      deck.shuffle()
      hand = deck.draw(n_cards)
      score = evaluator.evaluate([], hand)
      stats[score] += 1
    summation = np.cumsum(stats)
    summation /= summation[-1]
    print("-"*80)
    print("Table {0} of {1} cards known:".format(k, n_cards))
    for i in range(1, N_BINS):
      idx = int(i * N_SCORES / N_BINS)
      print("\tHand value {0} loses to {1}% of possible hands".format(idx, int(100*summation[idx])))

  sns.lineplot(x=np.arange(N_SCORES), y=stats / np.sum(stats))
  plt.savefig("dist_{}.png".format(n_cards))
  plt.clf()
  np.save("dist_{}.npy".format(n_cards), summation)

evaluate_table(5)
evaluate_table(6)
evaluate_table(7)
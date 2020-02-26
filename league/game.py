import numpy as np
import random

class GameEngine:
  def __init__(self, BATCH_SIZE):
    self.BATCH_SIZE = BATCH_SIZE

  def start_game(self, players):
    if len(players) != 6:
      raise ValueError("Only 6 players allowed")
    self.players = players
    random.shuffle(players)

    self.cards = np.tile(np.arange(52), (self.BATCH_SIZE, 1))
    for i in range(self.BATCH_SIZE):
      self.cards[i, :] = np.random.permutation(self.cards[i, :])
    self.cards = self.cards[:, :5 + 2 * len(players)]
import league.game
import numpy as np
import treys
import random
import time
from league.logger import GenericLogger
from agent.sac1.Sac1Agent import Sac1Agent
from agent.random.RandomAgentNP import RandomAgentNP


batch_size = 10000
game = league.game.GameEngine(BATCH_SIZE=batch_size, INITIAL_CAPITAL=100, SMALL_BLIND=2, BIG_BLIND=4, logger=GenericLogger())

players = [Sac1Agent(agent_id=1, config={'learning_rate': 0.01}), RandomAgentNP(), RandomAgentNP(), RandomAgentNP(),
           RandomAgentNP(), RandomAgentNP()]
iteration = 0
while True:
  print("Iteration", iteration)
  random.shuffle(players)
  t1 = time.time()
  scores = game.run_game(players)
  t2 = time.time()
  names = [str(p) for p in players]
  print("{0} games (computed in {2} seconds):\nMean: {1}\nMedian: {3}".format(batch_size,
                                                                              list(zip(names, scores.mean(axis=0).tolist())),
                                                                              t2 - t1,
                                                                              list(zip(names, np.median(scores, axis=0).tolist()))))
  iteration += 1
import league.game
import numpy as np
import treys
import random
import time
import json
from league.logger import GenericLogger
from agent.sac1.sac1Agent import Sac1Agent
from agent.random.randomAgentNP import RandomAgentNP


batch_size = 10000
game = league.game.GameEngine(BATCH_SIZE=batch_size, INITIAL_CAPITAL=100, SMALL_BLIND=2, BIG_BLIND=4, logger=GenericLogger())

with open(Sac1Agent._config_file_path(), 'r') as f:
  agents = json.load(f)['agent_ids']
ids = list(agents.keys())

players = [Sac1Agent(ids[0], agents[ids[0]]), RandomAgentNP(), RandomAgentNP(), RandomAgentNP(),
           RandomAgentNP(), RandomAgentNP()]
iteration = 0
while True:
  print("Iteration", iteration)
  random.shuffle(players)
  t1 = time.time()
  print("Simulating..")
  scores = game.run_game(players)
  t2 = time.time()
  names = [str(p) for p in players]
  print("{0} games (computed in {2} seconds):\nMean: {1}\nMedian: {3}".format(batch_size,
                                                                              list(zip(names, scores.mean(axis=0).tolist())),
                                                                              t2 - t1,
                                                                              list(zip(names, np.median(scores, axis=0).tolist()))))
  iteration += 1
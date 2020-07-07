from league.logger import NoneLogger
from league.agentManager import AgentManager
from strategy.compute_strat_vector import compute_strat_vector
import numpy as np
import os

if __name__ == '__main__':
  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json', models_path='./models')
  agent_manager.load()
  agent_ids = agent_manager.get_all_static_agent_ids()

  for idx,agent_id in enumerate(agent_ids):
    print("{0}%".format(round(100*idx/len(agent_ids), 2)))
    agent = agent_manager.get_instance(agent_id)
    if not os.path.exists(os.path.join("strategy", "strat_vectors", str(agent)+agent_id)):
      compute_strat_vector(agent, str(agent)+agent_id, verbose=False)
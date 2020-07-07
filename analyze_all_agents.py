from league.logger import NoneLogger
from league.agentManager import AgentManager
from strategy.compute_strat_vector import compute_strat_vector
import numpy as np

if __name__ == '__main__':
  agent_manager = AgentManager(file_path='./savefiles/agent_manager.json', models_path='./models')
  agent_manager.load()
  agent_ids = agent_manager.get_all_static_agent_ids()

  agent_id = agent_ids[0]
  agent = agent_manager.get_instance(agent_id)

  strats = np.stack((
    compute_strat_vector(agent, str(agent)+agent_id+"-1"),
    compute_strat_vector(agent, str(agent)+agent_id+"-2"),
    compute_strat_vector(agent, str(agent)+agent_id+"-3"),
    compute_strat_vector(agent, str(agent)+agent_id+"-4"),
  ))

  np.save("tmp.npy", strats)

  avgs = np.mean(strats, axis=-1)

  diffs = (
    avgs[0] - avgs[1],
    avgs[0] - avgs[2],
    avgs[0] - avgs[3],
    avgs[1] - avgs[2],
    avgs[1] - avgs[3],
    avgs[2] - avgs[3],
  )

  for diff in diffs:
    print(np.linalg.norm(diff))

  # for agent_id in agent_ids:
  #   agent = agent_manager.get_instance(agent_id)
  #   compute_strat_vector(agent, str(agent)+agent_id)
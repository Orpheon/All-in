import json
from collections import namedtuple
import numpy as np
import os
import random

from agent.qlearn1.qlearn1AgentNP import Qlearn1AgentNP
from agent.qlearn2.qlearn2AgentNP import Qlearn2AgentNP
from agent.qlearn3.qlearn3AgentNP import Qlearn3AgentNP
from agent.qlearn4.qlearn4AgentNP import Qlearn4AgentNP
from agent.random.randomAgentNP import RandomAgentNP
from agent.call.callAgentNP import CallAgentNP
from agent.sac1.sac1AgentNP import Sac1AgentNP
from agent.sac2.sac2AgentNP import Sac2AgentNP
from strategy.compute_strat_vector import compute_strat_vector

AGENT_TYPES = {'qln1': Qlearn1AgentNP,
               'qln2': Qlearn2AgentNP,
               'qln3': Qlearn3AgentNP,
               'qln4': Qlearn4AgentNP,
               'sac1': Sac1AgentNP,
               'sac2': Sac2AgentNP,
               'rndm': RandomAgentNP,
               'call': CallAgentNP}

AgentInfo = namedtuple('AgentInfo', ['AGENT_TYPE', 'MODEL_PATH', 'TRAINABLE', 'ORIGIN_DIVI', 'AGENT_NAME'])


class AgentManager:

  def __init__(self, file_path, models_path, possible_agent_names):
    self.FILE_PATH = file_path
    self.MODELS_PATH = models_path
    self.POSSIBLE_AGENT_NAMES = possible_agent_names
    self.POSSIBLE_AGENT_IDS = {'{:04}'.format(i) for i in range(10000)}  # TODO currently limited to 10000 agents
    self.agents = {}

  def add_agent(self, agent_type, trainable, origin_divi):
    taken_agent_ids = self.agents.keys()
    available_agent_ids = list(self.POSSIBLE_AGENT_IDS - taken_agent_ids)
    agent_id = random.choice(available_agent_ids)

    taken_agent_names = {a_info.AGENT_NAME for a_info in self.agents.values()}
    available_agent_names = list(self.POSSIBLE_AGENT_NAMES - taken_agent_names)
    agent_name = random.choice(available_agent_names)
    self.agents[agent_id] = AgentInfo(agent_type, '{}/{}'.format(self.MODELS_PATH, agent_id),
                                      trainable, origin_divi, agent_name)
    return agent_id

  def get_info(self, id) -> AgentInfo:
    return self.agents[id]

  def get_instance(self, agent_id, overwrite_not_trainable=False):
    agent = self.agents[agent_id]
    return AGENT_TYPES[agent.AGENT_TYPE].get_instance(agent.AGENT_TYPE, agent.MODEL_PATH,
                                                      not overwrite_not_trainable and agent.TRAINABLE,
                                                      agent_id, agent.ORIGIN_DIVI, agent.AGENT_NAME)

  def load(self):
    with open(self.FILE_PATH, 'r') as f:
      data = json.load(f)
    self.agents = {k: AgentInfo(*v) for k, v in data.items()}
    print('[AgentManager <- {}]: loaded {} agents'.format(self.FILE_PATH, len(self.agents)))

  def save(self):
    with open(self.FILE_PATH, 'w') as f:
      json.dump(self.agents, f, sort_keys=True, indent=2)
    print('[AgentManager -> {}]: saved {} agents'.format(self.FILE_PATH, len(self.agents)))

  def clone(self, agent_id, origin_divi):
    a_info = self.agents[agent_id]
    if a_info.TRAINABLE:
      instance = self.get_instance(agent_id)
      new_id = self.add_agent(a_info.AGENT_TYPE, False, origin_divi)
      new_path = self.agents[new_id].MODEL_PATH

      instance.clone_to_path(new_path)
      print('[AgentManager] cloned {} -> {}'.format(agent_id, new_id))
      return new_id
    return None

  def get_all_agents(self):
    return self.agents.items()

  def get_all_agent_types(self):
    return set(info.AGENT_TYPE for _,info in self.agents.items())

  def get_strategy_vector(self, id):
    root = os.path.join("strategy", "strat_vectors")
    path = os.path.join(root, id+"_strategy.npy")
    if os.path.exists(path):
      return np.load(path)
    else:
      print("Strategy vector of", id, "does not yet exist, computing.. (this may take a few minutes)")
      strategy, _ = compute_strat_vector(self.get_instance(id, overwrite_not_trainable=True))
      os.makedirs(root, exist_ok=True)
      np.save(path, strategy)
      return strategy

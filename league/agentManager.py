import json
from collections import namedtuple
import numpy as np

from agent.qlearn2.qlearn2AgentNP import Qlearn2AgentNP
from agent.qlearn3.qlearn3AgentNP import Qlearn3AgentNP
from agent.random.randomAgentNP import RandomAgentNP
from agent.call.callAgentNP import CallAgentNP
from agent.qlearn1.qlearn1AgentNP import Qlearn1AgentNP
from agent.sac1.sac1AgentNP import Sac1AgentNP
from agent.sac2.sac2AgentNP import Sac2AgentNP

AGENT_TYPES = {'qlearn1': Qlearn1AgentNP,
               'qlearn2': Qlearn2AgentNP,
               'qlearn3': Qlearn3AgentNP,
               'sac1': Sac1AgentNP,
               'sac2': Sac2AgentNP,
               'random': RandomAgentNP,
               'call': CallAgentNP}

AgentInfo = namedtuple('AgentInfo', ['AGENT_TYPE', 'MODEL_PATH', 'TRAINABLE'])


class AgentManager:

  def __init__(self, file_path, models_path):
    self.FILE_PATH = file_path
    self.models_path = models_path
    self.agents = {}

  def add_agent(self, agent_type, trainable):
    id = ''.join(str(i) for i in np.random.random_integers(0, 9, 8))
    self.agents[id] = AgentInfo(agent_type, '{}/{}'.format(self.models_path, id), trainable)
    return id

  def _add_agent_with_id(self, id, agent_type, trainable):
    self.agents[id] = AgentInfo(agent_type, '{}/{}'.format(self.models_path, id), trainable)

  def get_info(self, id) -> AgentInfo:
    return self.agents[id]

  def get_instance(self, id):
    agent = self.agents[id]
    return AGENT_TYPES[agent.AGENT_TYPE].get_instance(agent.TRAINABLE, agent.MODEL_PATH)

  def load(self):
    with open(self.FILE_PATH, 'r') as f:
      data = json.load(f)
    self.agents = {k: AgentInfo(*v) for k, v in data.items()}
    print('[AgentManager <- {}]: loaded {} agents'.format(self.FILE_PATH, len(self.agents)))

  def save(self):
    with open(self.FILE_PATH, 'w') as f:
      json.dump(self.agents, f, sort_keys=True, indent=2)
    print('[AgentManager -> {}]: saved {} agents'.format(self.FILE_PATH, len(self.agents)))

  def clone(self, id):
    agent = self.agents[id]
    if agent.TRAINABLE:
      instance = AGENT_TYPES[agent.AGENT_TYPE].get_instance(agent.TRAINABLE, agent.MODEL_PATH)

      new_id = ''.join(str(i) for i in np.random.random_integers(0, 9, 8))
      new_path = '{}/{}'.format(self.models_path, new_id)
      self.agents[new_id] = AgentInfo(agent.AGENT_TYPE, new_path, False)

      instance.clone_to_path(new_path)
      print('[AgentManager] cloned {} -> {}'.format(id, new_id))
      return new_id
    return None

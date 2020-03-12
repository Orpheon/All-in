import json
from pypokerengine.players import BasePokerPlayer


class BaseAgent(BasePokerPlayer):

  def __init__(self, agent_id, config):
    super().__init__()
    self.agent_id = agent_id
    self.config = config

  def __str__(self):
    err_msg = self._build_err_msg("__str__")
    raise NotImplementedError(err_msg)

  def get_name(self):
    return str(self)

  @classmethod
  def available_agent_ids(cls):
    with open(cls._config_file_path(), 'r') as f:
      config = json.loads(f.read())
      agent_ids = config['agent_ids'].keys()
    return agent_ids

  @classmethod
  def from_id(cls, agent_id):
    with open(cls._config_file_path(), 'r') as f:
      config = json.loads(f.read())['agent_ids'][agent_id]
      return cls(agent_id, config)

  @classmethod
  def _config_file_path(cls):
    err_msg = cls._build_err_msg("_config_file_path")
    raise NotImplementedError(err_msg)

  @classmethod
  def _build_err_msg(cls, msg):
    return "Your client does not implement [ {0} ] method".format(msg)

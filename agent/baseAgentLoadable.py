import json

from agent.baseAgentNP import BaseAgentNP


class BaseAgentLoadable(BaseAgentNP):

  def __init__(self, agent_id, config):
    super().__init__()
    self.agent_id = agent_id
    self.config = config

  def __str__(self):
    err_msg = self._build_err_msg("__str__")
    raise NotImplementedError(err_msg)

  def get_name(self):
    return str(self)

  def spawn_clone(self):
    pass

  @classmethod
  def available_agents(cls):
    with open(cls._config_file_path(), 'r') as f:
      config = json.loads(f.read())
    return [(agent_id, agent_info['matchup_info']) for agent_id, agent_info in config['agent_ids'].items()]

  @classmethod
  def from_id(cls, agent_id):
    with open(cls._config_file_path(), 'r') as f:
      config = json.loads(f.read())
    setup = config['agent_ids'][agent_id]['setup']
    return cls(agent_id, setup)

  @classmethod
  def retire(cls, agent_id):
    with open(cls._config_file_path(), 'rw') as f:
      config = json.loads(f.read())
      config['agent_ids'][agent_id]['setup']['type'] = 'retired'
      f.write(json.dumps(config))

  @classmethod
  def _config_file_path(cls):
    err_msg = cls._build_err_msg("_config_file_path")
    raise NotImplementedError(err_msg)

  @classmethod
  def _build_err_msg(cls, msg):
    return "Your client does not implement [ {0} ] method".format(msg)

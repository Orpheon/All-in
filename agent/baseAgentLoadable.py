import json

from agent.baseAgentNP import BaseAgentNP


class BaseAgentLoadable(BaseAgentNP):

  def __init__(self, agent_id, config):
    super().__init__()
    self.agent_id = agent_id
    self.config = config

  def __str__(self):
    if not self.config['setup']['trainable']:
      return self.agent_id
    else:
      return '{}-GEN{:03}'.format(self.agent_id, self.config['setup']['generation'])

  def save_config(self):
    with open(self._config_file_path(), 'r') as f:
      config_data = json.load(f)
    config_data['agent_ids'][self.agent_id] = self.config
    with open(self._config_file_path(), 'w') as f:
      json.dump(config_data, f, indent=2, sort_keys=True)

  @classmethod
  def available_agents(cls):
    with open(cls._config_file_path(), 'r') as f:
      config = json.loads(f.read())
    return [(agent_id, agent_info) for agent_id, agent_info in config['agent_ids'].items()]

  @classmethod
  def from_id(cls, agent_id):
    with open(cls._config_file_path(), 'r') as f:
      all_config = json.loads(f.read())
    config = all_config['agent_ids'][agent_id]
    return cls(agent_id, config)

  @classmethod
  def _config_file_path(cls):
    err_msg = cls._build_err_msg("_config_file_path")
    raise NotImplementedError(err_msg)

  @classmethod
  def _build_err_msg(cls, msg):
    return "Your client does not implement [ {0} ] method".format(msg)

import os
import shutil


class BaseAgentNP:
  MODEL_FILES = []

  def __init__(self, agent_type, model_path, trainable, agent_id, origin_league, agent_name):
    self.AGENT_TYPE = agent_type
    self.MODEL_PATH = model_path
    self.TRAINABLE = trainable
    self.AGENT_ID = agent_id
    self.ORIGIN_LEAGUE = origin_league
    self.AGENT_NAME = agent_name

  @classmethod
  def get_instance(cls, agent_type, model_path, trainable, agent_id, origin_league, agent_name):
    return cls(agent_type, model_path, trainable, agent_id, origin_league, agent_name)

  def __str__(self):
    return '{}-{}.{}{}-{}'.format(self.AGENT_TYPE, self.ORIGIN_LEAGUE, self.AGENT_ID, 'S' if self.TRAINABLE else 'T',
                                  self.AGENT_NAME)

  def act(self, player_idx, round, active_rounds, current_bets, min_raise, prev_round_investment, folded, last_raiser,
          hole_cards, community_cards):
    err_msg = self._build_err_msg('act')
    raise NotImplementedError(err_msg)

  def initialize(self, batch_size, initial_capital, n_players):
    pass

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser,
                     hole_cards, community_cards, gains):
    pass

  def load_model(self):
    pass

  def save_model(self):
    pass

  def clone_to_path(self, new_path):
    os.makedirs(new_path, exist_ok=True)
    for file in self.MODEL_FILES:
      src = os.path.join(self.MODEL_PATH, file)
      dst = os.path.join(new_path, file)
      shutil.copyfile(src, dst)

  @classmethod
  def _build_err_msg(cls, msg):
    return "Your client does not implement [ {0} ] method".format(msg)

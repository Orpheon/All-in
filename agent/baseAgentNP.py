import os
import shutil


class BaseAgentNP:
  MODEL_FILES = []

  def __init__(self, trainable, model_path, agent_id):
    self.trainable = trainable
    self.model_path = model_path
    self.agent_id = agent_id

  @classmethod
  def get_instance(cls, trainable, model_path, agent_id):
    return cls(trainable, model_path, agent_id)

  def __str__(self):
    return self.agent_id

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
      src = os.path.join(self.model_path, file)
      dst = os.path.join(new_path, file)
      shutil.copyfile(src, dst)

  @classmethod
  def _build_err_msg(cls, msg):
    return "Your client does not implement [ {0} ] method".format(msg)

class BaseAgentNP:

  def __init__(self):
    pass

  def act(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards):
    err_msg = self._build_err_msg('act')
    raise NotImplementedError(err_msg)

  def start_game(self, batch_size, initial_capital, n_players):
    pass

  def end_trajectory(self, player_idx, round, current_bets, min_raise, prev_round_investment, folded, last_raiser, hole_cards, community_cards, gains):
    pass

  @classmethod
  def _build_err_msg(cls, msg):
    return "Your client does not implement [ {0} ] method".format(msg)

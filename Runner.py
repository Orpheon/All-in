import random

from pypokerengine.players import BasePokerPlayer

from agent.BeginnerPlayer import BeginnerPlayer
from baseline.BaselinePokerPlayer import BaselinePlayer
from baseline.CallBaselinePokerPlayer import CallBaselinePlayer
from baseline.RandomPokerPlayer import RandomPlayer
from configuration.CashGameConfig import CashGameConfig
from league.Rating import Rating


class Runner:

  _all_players = [BeginnerPlayer(0.5, 0.5),
                  BeginnerPlayer(0.5, 0.3),
                  BeginnerPlayer(0.5, 0.0),
                  BeginnerPlayer(0.3, 0.0),
                  BeginnerPlayer(0.0, 0.0),
                  #BaselinePlayer(),
                  CallBaselinePlayer(),
                  RandomPlayer()]
  _rating = Rating('./runner_placings.json')

  def run_games(self, hands):
    poker_config = CashGameConfig(evaluations=hands)

    players = self.generate_matchup()
    for player in players:
      poker_config.register_player(player.id, player)
    placings = poker_config.run_evaluation()
    self._rating.update_from_placings(placings)

  def generate_matchup(self):
    #TODO: make not hard coded
    return random.sample(self._all_players, 6)

if __name__ == '__main__':
  runner = Runner()
  for i in range(20):
    runner.run_games(20000)
  runner._rating.plot_history()

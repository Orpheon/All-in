import league.game

game = league.game.GameEngine(5)
game.run_game([0, 0, 0, 0, 0, 0])
print(game.community_cards)
# All-In #

All-In is a poker AI developed for a bachelor thesis by
[Jens Kaminsky](https://github.com/BBBot2015) and
[Alex Liebendörfer](https://github.com/Orpheon).

The accompanying paper can be found
[here](https://github.com/Orpheon/Pokerbot-Thesis/blob/master/Pokerbot_Thesis.pdf).

This code was written specifically for that paper is neither particularly maintained
nor documented. The proper way to launch a simulation is to execute `runner-py` after
tuning it to desire. All parameters and data about agents are stored in `savefiles/`
and agent weights are stored in `models/`. Plots and strategy vectors can be computed
by executing `plot_existing_strategies.py` and `agent_aggregator.py` after modification.

This respository was seeded off of
[PyPokerEngine Competition Starter Kit](https://github.com/YanickSchraner/pokerchallenge)
and heavily uses code from [SpinningUp](https://github.com/openai/spinningup).
We are grateful for both.
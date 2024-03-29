# All-In #

All-In is a poker AI developed for a bachelor thesis by
[Jens Kaminsky](https://github.com/BBBot2015) and
[Anya Liebendörfer](https://github.com/Orpheon).

The accompanying paper can be found
[here](https://github.com/Orpheon/Pokerbot-Thesis/blob/master/Pokerbot_Thesis.pdf).

This code was written specifically for that paper and is neither particularly maintained
nor documented. The proper way to launch a simulation is
to execute `runner.py` after tuning it to desire. All parameters and data about agents
are stored in `savefiles/` except for agent weights which are stored in `models/`. Plots and strategy
vectors can be computed by executing `plot_existing_strategies.py` and `agent_aggregator.py`
after modification and after agents have been trained.

This repository was seeded off of
[PyPokerEngine Competition Starter Kit](https://github.com/YanickSchraner/pokerchallenge)
and heavily uses code from [SpinningUp](https://github.com/openai/spinningup).
We are grateful for both, and all credit for their respective code goes to them.

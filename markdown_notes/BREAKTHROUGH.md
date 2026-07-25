(15.06.2026) Cycle 186 won against previous best Quoridor AI https://gorisanson.github.io/quoridor-ai/ as 2nd player at 2000 MCTS sims (roughly the same compute time for both players).
It was winning as 1st player way before.
Need to play 186 as 1st player to confirm.
Also try and play some games at non zero temperature, e.g. temp=0.15 to see how stable it is and how often can it still win by sometimes deviating from its argmax move selection.

(16.07.2026) Cycle 234 also wins as P2 at 2000 sims.
It also won at 1600 sims but lost at 800 or 1200.

(17.07.2026) Cycle 339 wins as P2 at 1600 sims. Still loses at 1200 and 800 though.

(19.07.2026) After the reversed policy fix, cycle 359 wins as P2 (and P1) at 800 sims!
It even wins at 400 sims! Even at 200 sims it still won!

(20.07.2026) After the heads redesign and supervised training on latest gathered data, a the NN came out so good, that it wins as P1 and P2 without MCTS! (equivalent to 1 sim, only the NN policy argmax played in each position)
It won't get much better than that i guess. I mean it could, but there will be no way to evaluate progress besides self-play against older checkpoints. Since I can't beat the agent already and neither can the eval gorisanson bot.

(25.07.2026) The fresh from-scratch 9x9 run finally beat the old best (heads cycle 56).
Cycle 244 wins 63.8% h2h over 500 games (315W-8D-177L) and takes rank 1 in the v6 tournament (100 sims, temp 0.5).
First crossover was cycle 218 at ~53%, now 244 is at 63.8%. Still climbing: 65.5% vs 218, 68.8% vs 207.

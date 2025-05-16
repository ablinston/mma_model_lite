# Feature ideas

* Add odds accuracy history for that fighter

# To do scripts


* 2025-05-15 - Issues with Betfair odds think resolved, but some oddities. Would it be better to take the average of the last 3 trades, rather than just the last trade before 5pm?
* 45% of the Betfair odds are missing, through non-"UFC" events. Are things like DWCS included in the training data?

* Betfair downloaded data only has the market info, but see if we can add available offers. The live pricing might help reduce the 20% missing data for the odds.
Looking at (c:/Users/Andy/Desktop/1.241955374) which is missing in the processed odds data. Only has bets for a limited number of dates because I think it was a last minute replacement. But it was announced on April 3, but maybe there where no bets until later?
* 

* Change the ELO parameter choices so that it accounts for the scale of how wrong the ELO is, not just predicting binary. Turn ELO into a probability and use logloss?

* Add test scripts of the backtesting and for the odds to check it's working correctly.

* Add a manual lookup list for betfair odds fighter names where there are small differences


# Conclusions

Each time I've tested removing split decisions or non W/L, L/W the AUC gets worse on test data including all results.
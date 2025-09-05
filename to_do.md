# Feature ideas

* Add odds accuracy history for that fighter
* Add adjusted accuracy like with the bayes per minute adjusted stats

# To do scripts

* Add automated check of the data stats log against last entry before save?

* Betfair downloaded data only has the market info, but see if we can add available offers. The live pricing might help reduce the 20% missing data for the odds.
Looking at (c:/Users/Andy/Desktop/1.241955374) which is missing in the processed odds data. Only has bets for a limited number of dates because I think it was a last minute replacement. But it was announced on April 3, but maybe there where no bets until later?
* 

* Change the ELO parameter choices so that it accounts for the scale of how wrong the ELO is, not just predicting binary. Turn ELO into a probability and use logloss?

* Add a manual lookup list for betfair odds fighter names where there are small differences


# Conclusions

Each time I've tested removing split decisions or non W/L, L/W the AUC gets worse on test data including all results.
# Feature ideas

* Add odds accuracy history for that fighter

# To do scripts


* 2025-05-02 Am up to adding the upcoming bouts to the clean data function. Then need to finish adding some checks to the check_data_coverage function

* Betfair downloaded data only has the market info, but see if we can add available offers. The live pricing might help reduce the 20% missing data for the odds.
Looking at (c:/Users/Andy/Desktop/1.241955374) which is missing in the processed odds data. Only has bets for a limited number of dates because I think it was a last minute replacement. But it was announced on April 3, but maybe there where no bets until later?
* Proposed changes: change so that it checks for the opponent and the surname of the fighter for a match. If both the surnames are in the names then give it a match. Also, where there is no data for a fight before the cutoff, just take the earliest bet?

* Change the ELO parameter choices so that it accounts for the scale of how wrong the ELO is, not just predicting binary. Turn ELO into a probability and use logloss?

* Add test scripts of the backtesting and for the odds to check it's working correctly.

* Add a manual lookup list for betfair odds fighter names where there are small differences


# Conclusions

Each time I've tested removing split decisions or non W/L, L/W the AUC gets worse on test data including all results.
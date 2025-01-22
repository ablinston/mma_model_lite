# Feature ideas

* Add odds accuracy history for that fighter

# To do scripts

* Merge odds based on date +- 2 as well as date +- 1

* Change the ELO parameter choices so that it accounts for the scale of how wrong the ELO is, not just predicting binary. Turn ELO into a probability and use logloss?

* Fix aggregation error in script 6

* Add test scripts of the backtesting and for the odds to check it's working correctly.
* Change test split so it can exclude test fighters instead of fights specifically.

* Add sample sizes to the metric outputs in the test scripts
* Fix error with missing groups in script 10 functions
* Add a manual lookup for fighters odds urls, and get it to handle people like Volkanovski that have 2 pages

* Fill in missing dates with a very early one. To avoid missing in diff where it could be either fighter that hasn't been KO'd for e.g.

# Conclusions

Each time I've tested removing split decisions or non W/L, L/W the AUC gets worse on test data including all results.
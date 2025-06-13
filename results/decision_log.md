# Fight history

Based on the test results of the first xg boost forest, on the test dataset, the model performed strongest when looking at fighters with both having >3 fights. The OOT showed the best AUC result in this group also, so it makes sense to only train on this data.

Experiment: http://localhost:8080/#/experiments/490730489596532803/runs/78d44d90d4224a1fb9ee2f3f1f017815

Have tried this again with newer model and it has a negative ROI on fights without both > 2 fights.

# Data parameters

* Betfair imputation: although 50% of records have Betfair data missing, only 5% of the final training data ends up being imputed because the missing events are outside the usual training sample.
* Test/Train split odds loglos: Have found that it's best to have similar odds logloss in test and train so it can see overfitting.

# Betting parameters

Version 2 XGB Forest model shows much better correlation between EV predictions and actuals profits and losses in bet testing v15 in MLflow. Highest P&L was with 0 and 0 min probs and EVs (both in OOT and Test), but max drawdown was too large. 0.49 and 0 as prob and EV mins was the next highest overall P&L for both OOT and Test with much more tolerable drawdowns.

Don't place bets to get filled unless the Betfair odds at 5pm GMT give a positive bet signal. The OOT testing I did showed that they are negative ROI and at best they will be low EV bets.

# Modelling parameters

* Split decisions and other results:
    * **Don't remove**: Reduced the fit of the models when excluded, 2014+_gt2_fights_xgboost_forest_v10. Still a 0.04 difference in logloss (train better than test)
* Column proportion in xgboost forest: 
    * **Use 1**: found that using all the columns in every model significantly decreased logloss. Needed much fewer trees. But logloss difference between test and train  rose to around 0.3, but test had a better logloss! Also seen that on OOT data.
* True odds only: 
    * **Use?**: I found that using this led to slightly better fit on test data but worse on training data. This brought the two loglosses closer together though.
* Bayes features:
    * **Don't use**: found the logloss was worse after adding these. The EV MSE was worse for both test and train but they were both closer together. Even with feature trimming, it didn't improve more than features without bayes.
* Feature trimming:
    * Testing when looking at the bayes variables and it improved the model fit.
* SHAP importance from xgboost:
    * Keep ussing random forest. Models didn't fit as well
* col subsample in xgboost:
    * Using the col subsample within xgboost resulted in slightly better metrics in CV. 
* bagging in xgboost forest:
    * **enable**: Saw a significant improvement in logloss and EV MSE when enabling bagging to increase sample sizes in v15. But the model did go from doing slightly better in logloss in the test data to having slightly worse logloss in test data than train, but still better overall logloss in test.
* regularisation:
    * Didn't improve the forest fit in v16, but didn't see any evidence of overfitting so that makes sense.
* min bucket size:
    * **keep at 15**: Tried making it variable but seem to lead to some overfitting
* Randomising odds in training data:
    * **Don't use**: it didn't improve overall fit, didn't really reduce variance when testing randomised odds. Just ensure to do the sensitivity testing for the model to check what possible returns would have been.
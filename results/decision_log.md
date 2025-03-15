# Fight history

Based on the test results of the first xg boost forest, on the test dataset, the model performed strongest when looking at fighters with both having >3 fights. The OOT showed the best AUC result in this group also, so it makes sense to only train on this data.

Experiment: http://localhost:8080/#/experiments/490730489596532803/runs/78d44d90d4224a1fb9ee2f3f1f017815

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
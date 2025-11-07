# mma_model
A model to predict MMA fight results.

To use this, open a conda terminal, navigate to the folder and use `conda activate mma_model`. You can then run `spyder` to use Spyder IDE.

# Updating training data

The full data scrape code is found in the `src` folder. Scripts 1, 2 and 3 will perform a full scrape of the data and all fighters stats.

Once this has been done, script 4 can be run to add new events and fighter stats to the existing data. **To update, only script 4 needs to be run.**

# Bayesian Beta Binomial: The Zero Division Solution

When calculating success rates (like submission accuracy or strike accuracy), we use Bayesian Beta Binomial analysis to provide meaningful priors that smoothly handle edge cases.

The approach works like this: Instead of naive division that breaks on zeros, we model each ratio as a Beta distribution where:

```
α (alpha) represents successful attempts plus prior successes
β (beta) represents failed attempts plus prior failures
The posterior mean (α / (α + β)) gives us our smoothed estimate
```

For example, with submission accuracy:

`submission_accuracy = (submissions + α) / (submission_attempts + α + β)`

α is the total submissions from the last n fights and α + β is the total attempts at submissions. In my functions, these are scaled by a factor called `bayes_alpha_plus_beta_rebase` for both per minute stats and percentage stats. This is to stop them dominating the formula. They will be rebased to a number of total attempts or minutes depending on what's being calculated.

For example, if there are 100 submission attempts across 1000 minutes in the history, this could be rebased to give α = 1, α + β = 10.

## Unit testing

There are some unit test scripts saved within `./tests/functions/`. To run this, open an anaconda prompt, activate the `mma_model` environment and then change directory to `./tests/functions/`. Then use command `pytest`.

# Training Experiments

These use MLflow, so ensure the server is running using (or use the .bat file in the root repo)
`mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri /Documents/mma_model/mlruns/`

I couldn't get this to work when running in venv, so has to be run outside. Just ensure that a consistent version of mlflow is used inside and outside.

## Metric choice

I chose AUC to assess the models and target rather than accuracy. Although we are only interested in predicting the winners, if we want to make money then the odds/probability of each outcome is the most important thing to get correct.

# GPU

Training of models runs much faster on GPU than CPU. There is a test script in the tests folder to verify this.

The experiments run with GPU on an Nvidia GTX 1060. To use this, lightgbm needs installing in a specific way. Follow the instructions at https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#id18

It needs a C++ compiler installed (minGW is how to get it on Windows https://sourceforge.net/projects/mingw/). The gcc is the fidliest to install. Needs to install the gcc part with MinGW and then manually add the path.
```
pip install --upgrade cmake

pip install --no-binary lightgbm --config-settings=cmake.define.CMAKE_C_COMPILER=C:/MinGW/bin/gcc --config-settings=cmake.define.CMAKE_CXX_COMPILER=C:/MinGW/bin/g++ --config-settings=cmake.define.PYTHON_INCLUDE_DIR=C:/Users/Andy/AppData/Local/Programs/Python/Python311/include --config-settings=cmake.define.PYTHON_LIBRARY=C:/Users/Andy/AppData/Local/Programs/Python/Python311/libs/python311.lib --config-settings=cmake.define.USE_CUDA=ON lightgbm

```
AB note: couldn't get GPU training with LightGBM working at all on Windows. Used conda and xgboost instead (for random forests too because it allows GPU training).

```
conda install -c nvidia cuda-toolkit==11.8 # HAVE TO GET THE RIGHT VERSION THAT's STABLE WITH GTX 1060 
conda install -c conda-forge py-xgboost=*=cuda* --force-reinstall
# or
pip install --force-reinstall --no-deps xgboost==2.1.4
```

# Notes

To undo stupid split in Spyder, use Alt + Shift + W

## Odds

Decimal odds are how much is received back for every 1 bet. So for example 2 means you receive 2 (original bet + 1 profit)

## Betfair API

The Betfair API is accessed through the `betfairlightweight` python package. It is relatively complex to navigate with little documentation.

Betfair historical data needs to have packages purhcased manually on the website https://historicdata.betfair.com/#/home. Seems to be just needed on a calendar month basis which will then allow files to be downloaded during that month.

Historic data has too many files so better to download online then extract the bz2 files.

MMA is lumped with everything else so the scripts scan the files and see which ones are fights. Each file is a `market` which is the market for a specific outcome e.g. Fight Result for Conor vs Khabib. The package creates `MarketBook`s which are the details of a market, and `RunnerBook`s which are betting on a specific outcome e.g. Conor to win.

## Playwright

To install, need to install the package then install the browsers with `python -m playwright install
` 

# DVC Setup

Set up, using Backblaze (https://secure.backblaze.com/b2_buckets.htm) with S3 is a bit of a nightmare. FIrst need to install dvc that is compatable, then there are some recent updates that don't allow dvc to push properly so need older versions of packages.

```
pip install dvc[s3]
pip install boto3==1.35.90
pip install aiobotocore==2.16.1
pip install botocore==1.35.88
```

Or, I've succcessfully installed it with conda with less trouble.

```
conda install dvc[s3]
conda install dvc-s3
# or 
conda install -c conda-forge dvc-s3 --update-deps --force-reinstall
```
Had a common issue with a '128m' error. If this happens go in the .git/config file and comment out the limits in the `[pack]` section.

Then need to set up the remote. I have called it `storage`.

```
dvc remote add -d storage s3://mma-model-lite
dvc remote modify storage endpointurl https://
s3.eu-central-003.backblazeb2.com
dvc remote modify storage access_key_id YOUR_KEY_ID_HERE
dvc remote modify storage secret_access_key YOUR_APP_KEY_HERE
```

Push all git commits for dvc data with `dvc push -r storage --all-commits`. Otherwise, dvc will just push the datasets from the active commit. (doesn't seem to work right now so push frequently. Might start working if the latest version of DVC finally works with Backblaze.)

## Clearing cache

It's good to clear the remote cache and delete datasets in deleted branches. Most live updates should be deleted as they aren't needed and dvc commits to main minimised.
 
To clear the remote cache, run

```
# Ensure all commits local
dvc pull --all-branches --all-tags --run-cache

# Show what will be deleted
dvc gc --workspace --cloud --dry

# Delete, but keeping all commits from all branches
dvc gc --all-branches --all-tags --cloud

```
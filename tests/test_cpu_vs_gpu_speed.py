import itertools
import lightgbm as lgb
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

# Read in global objects and packages
exec(open('global.py').read())

# Read in the data for training
data = pq.ParquetDataset('./data/final_dataset/').read().to_pandas()

data.replace([np.inf], 99999, inplace = True)
data.replace([-np.inf], -99999, inplace = True)

# Remove column that has results
data.drop(columns = ['method', 'fighter1', 'fighter2'], inplace = True)

# Set target
target = 'outcome'

# Recode the target. Don't need to distinguish between D and NC
data[target] = np.where((data[target] == "D/D") | (data[target] == "NC/NC"), "OTH", data[target])

# Convert all dates to integers
date_cols = [col for col in data.columns if data.dtypes[col] == 'datetime64[ns]']
data[date_cols] = data[date_cols].astype('int64')

# Convert object columns into series' of dummies
obj_cols = [col for col in data.columns if data.dtypes[col] == 'object']
obj_cols.remove(target)
data = pd.get_dummies(data, columns = obj_cols)

# Convert the target into a labelled column ready for multiclass modelling
le = LabelEncoder()
data[target] =le.fit_transform(data[target])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns = [target]),
    data[target],
    test_size = 0.1,
    random_state = config['seed_all']
)

# Set number of classes for training
no_of_classes = len(set(y_train))

# Set the number of max bins to use. Default is 255
max_bins = 63

# Set up experiment training
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment('cpu_vs_gpu_test')


# Format the data for the model
train = lgb.Dataset(X_train, label = y_train)

# Run the test using the CPU
with mlflow.start_run():

    params = {
        'objective': 'multiclass',      # Binary classification
        'metric': 'multi_logloss', # Evaluation metric
        'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
        'num_leaves': 31,           # Controls the complexity of the model
        'learning_rate': 0.01,      # Smaller learning rate for better accuracy
        'feature_fraction': 0.9,    # Subsampling of features to speed up training
        'bagging_fraction': 0.8,    # Subsampling of data to prevent overfitting
        'bagging_freq': 5,          # Perform bagging every 5 iterations
        'n_jobs': 3,               # Use all CPU cores
        'verbose': -1,            # Suppress output
        'num_class': no_of_classes,
        'num_iterations': 200
    }

    lightgbm_model = lgb.train(params, train)

    mlflow.log_param('model', 'LightGBM')
    mlflow.log_params(params)

# # Run the same model using the GPU

## AB: Couldn't get LightGBM to actually use the GPU

with mlflow.start_run():

    params.update({
        'device': 'cuda',  # Use GPU
        'gpu_platform_id': 0,  # Platform ID, 0 for default
        'gpu_device_id': 0,  # Device ID, 0 for the first GPU
    })

    lightgbm_model = lgb.train(params, train)

    mlflow.log_param('model', 'LightGBM')
    mlflow.log_params(params)


# Run the same test with xgboost

train = xgb.DMatrix(X_train, label = y_train)

with mlflow.start_run():

    params = {
        'max_depth': 5,                 # Maximum depth of a tree
        'eta': 0.01,                     # Learning rate
        'objective': 'multi:softprob', # Binary classification objective
        'eval_metric': 'mlogloss',       # Evaluation metric
        'tree_method': 'hist',      # Use GPU accelerated algorithm
        'verbosity': 1,
        'num_class': no_of_classes,
        'n_folds': 5,                   # Number of folds for the cross-validation
        'random_seed': config['seed_all'],
        'num_iterations': 200
    }

    model = xgb.train(params, train, num_boost_round = params['num_iterations'])

    mlflow.log_param('model', 'XGBoost')
    mlflow.log_params(params)

with mlflow.start_run():
    
    params.update({
        'device': 'cuda',           # Use CUDA / GPU
        'predictor': 'gpu_predictor',   # Use GPU for prediction
    })

    model = xgb.train(params, train, num_boost_round = params['num_iterations'])

    mlflow.log_param('model', 'XGBoost')
    mlflow.log_params(params)

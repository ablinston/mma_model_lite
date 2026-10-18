# FastAPI service over the XGBoost ensemble. Loads the model and the upcoming card once at startup,
# then scores a single bout against odds supplied by the caller.

from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import fastparquet as fp
import joblib
import json
import logging
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from secrets import compare_digest
import sys
import uuid

# These can import by name because serving sits outside the circular references between the mma_functions modules
from mma_functions.config import data_processing_config, model_hyper_params
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.features import add_fighter_differentials
from mma_functions.modelling import predict_xgboost_ensemble
from mma_functions.odds import calculate_all_true_odds_columns

logger = logging.getLogger(__name__)

# Define the context var for the Request ID, to be set later
request_id_var = ContextVar('request_id_var', default='-')

################################################################################
### Request and response shapes, and settings ###
################################################################################

# The body posted to /predict. Pydantic validates it and returns a 422 before the endpoint runs if anything fails
class FightRequest(BaseModel):

    # Reject unknown fields rather than ignoring them, so a misspelled field name gets reported instead of quietly dropped
    model_config = ConfigDict(extra = 'forbid')
    # Pydantic will create the __init__ and attributes from these annotations as well as the validator , and OpenAPI schema
    bout: str
    # Decimal odds of 1.0 are a certainty, so anything at or below that is impossible and gets refused
    fighter1_odds: float = Field(gt = 1.0)
    fighter2_odds: float = Field(gt = 1.0)

# What /fights returns per fight. Declaring it means only these fields can leave, whatever else is on the frame
class FightSummary(BaseModel):

    bout: str
    date: datetime
    event: str
    fighter1: str | None
    fighter2: str | None

# Set up the json formatter for the logs
class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Build the dictionary for the log
        log = {'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
         'request_id': request_id_var.get(),
         'level': record.levelname,
         'logger': record.name,
         'message': record.getMessage()}

        if record.exc_info:
            log['exception'] = self.formatException(record.exc_info)

        return json.dumps(log)

# Define the settings, which are used to configure the API such as model version etc
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = Path(__file__).parent / '.env')
    ensemble_path: str # These will come automatically from the environment ENSEMBLE_PATH, from an .env file like in docker or Kubernetes
    dataset_path: str
    log_level: str = 'INFO'
    auth_token: str

# The environment gets read here at import, so a missing variable stops the process now rather than on the first request
settings = Settings()
bearer_scheme = HTTPBearer(auto_error = False) # Makes a missing header return the raised Exception rather than default 403

# Work out which odds column the model was trained on once here, rather than writing back to the config inside a request
if data_processing_config['impute_missing_betfair'] and model_hyper_params['odds_to_use'] == 'betfair_exch_odds':
    odds_to_use = 'betfair_exch_odds_imputed'
else:
    odds_to_use = model_hyper_params['odds_to_use']

# Setup logging handler
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
# Set this logging on the root logger. This is the final handler applied (after the serving.main handler) so will apply it for all our logs
logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)

################################################################################
### Security ###
################################################################################

# Verification function for endpoint authorisation
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):

    # Verify using compare_digest, which always takes a fixed time so avoids issues of solving token by time taken to match
    if credentials is None:
        raise HTTPException(status_code = 401, detail = "Authentication credentials missing")
    else:
        if not compare_digest(credentials.credentials, settings.auth_token):
                    raise HTTPException(status_code = 401, detail = "Authentication failed")


################################################################################
### Startup ###
################################################################################

# Define the startup actions in the lifespan. Use async as this is only run once and this is needed as FastAPI is an ASGI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the XGBoost ensemble
    app.state.ensemble = joblib.load(settings.ensemble_path) # Can use settings object here without as an input because it is imported
    logger.info('Loaded model %s', settings.ensemble_path)

    # Load the upcoming fights data
    full_data = fp.ParquetFile(settings.dataset_path).to_pandas()
    app.state.fights = full_data[full_data['outcome'].isna()].copy()

    # Record when the dataset was last written, so /ready can report how stale the card is
    app.state.data_modified = datetime.fromtimestamp(Path(settings.dataset_path).stat().st_mtime).isoformat()
    logger.info('Data loaded from %s, last modified: %s', settings.dataset_path, app.state.data_modified)
    del full_data

    # Everything above runs before the server accepts a request. Anything after the yield would run on shutdown
    yield

# Build the app and register the startup hook. This has to come before the endpoints, which register themselves against it
app = FastAPI(lifespan = lifespan)


################################################################################
### Middleware ###
################################################################################

# Runs before every path request
@app.middleware("http")
async def assign_id_to_context(request: Request, call_next):

    # Get the id from the request header
    u_id = request.headers.get('X-Request-Id') or str(uuid.uuid4())

    # Set the id in context
    request_id_var.set(u_id)

    # Get the response from the app functions/endpoint
    response = await call_next(request)

    # Set the same ID in the headers of the response to report to the user for debugging purposes
    response.headers['X-Request-Id'] = u_id

    return response

################################################################################
### Endpoints ###
################################################################################

@app.get('/')
def placeholder():
    pass
# Emoty endpoint for now

# Liveness, meaning the process is alive and nothing more. A failure here restarts the pod, so it must not check anything a restart would not fix
@app.get('/health')
def health_check():
    return {'status': 'ok'}

# Readiness, meaning there is something worth serving. A 503 here takes the pod out of the load balancer but leaves it running
@app.get('/ready')
def ready_check(request: Request):

    ready_check = {}

    # Load in the required objects for predictions
    fights_data = request.app.state.fights
 
    # Refuse traffic once the whole card is in the past, because every prediction would be for a fight already settled
    if fights_data['date'].max().date() < datetime.today().date():
        raise HTTPException(status_code = 503, detail = 'Fight data is in the past')

    missing_bouts = int(fights_data['bout'].isna().sum())
    
    ready_check.update({
        'missing_bouts': missing_bouts,
        'servable_fights': len(fights_data) - missing_bouts})

    ready_check.update({
        'model_version': Path(settings.ensemble_path).name,
        'data_modified': request.app.state.data_modified})

    ready_check.update({'status': 'ready'})

    return ready_check
    
# Prediction end point, authorised using fn verify_token
@app.post('/predict', dependencies = [Depends(verify_token)])
def predict_fight(fight: FightRequest, request: Request):

    # Log the prediction call
    logger.info('Predicting %s', fight.bout)

    # Copy the card, because the frame is shared between requests and the functions below write to whatever they are given
    card = request.app.state.fights.copy()

    # Find the bout
    mask = card['bout'] == fight.bout

    # Raise exception if empty
    if not mask.any():
        raise HTTPException(status_code = 404, detail = f'No upcoming fight for bout {fight.bout}')
    elif mask.sum() > 1:
        raise HTTPException(status_code = 500, detail = f'Duplicate fight for bout {fight.bout} found in data')

    # Replace the odds
    card.loc[mask, 'fighter1_betfair_exch_odds'] = fight.fighter1_odds
    card.loc[mask, 'fighter2_betfair_exch_odds'] = fight.fighter2_odds

    # Recalculate the true odds off the new odds, for the whole card
    card = calculate_all_true_odds_columns(
        card,
        impute_missing_betfair = data_processing_config['impute_missing_betfair'],
        odds_suffix_for_true_odds = data_processing_config['upcoming_odds_suffix_for_true_odds'],
        impute_one_sided_missing_odds = data_processing_config['impute_one_sided_missing_odds'])

    # Set the odds columns the model was trained on
    card['fighter1_backtesting_odds'] = card['fighter1_' + odds_to_use].copy()
    card['fighter2_backtesting_odds'] = card['fighter2_' + odds_to_use].copy()

    # Recalculate the differentials. This comes back through polars with a new index, so find the bout again rather than reusing the mask
    card = add_fighter_differentials(card)
    row = card[card['bout'] == fight.bout]

    # Note the index label of the requested bout now, because prep turns the bout column itself into dummies
    bout_label = row.index[0]

    # Prep the whole card so the dummy columns match what training produced. Only the first item is needed, the rest are train and test splits that are empty for upcoming fights
    X = prep_data_for_model_training(card, upcoming_fights = True)[0]

    # Rename to match the model's own feature names before the reindex, the same way predict_xgboost_ensemble does internally
    ensemble = request.app.state.ensemble
    if any('_sub.att' in feature for feature in ensemble[0]['features']):
        X.columns = X.columns.str.replace('_sub_attempt', '_sub.att')

    # Force the exact columns and order the model expects, filling absent dummy levels with zero
    expected = ensemble[0]['model'].feature_names
    filled = len([col for col in expected if col not in X.columns])
    X = X.reindex(columns = expected, fill_value = 0)

    # A large number here means more is out of step than a few absent dummy levels
    if filled > 0:
        logger.warning('%s of %s model features were missing from the prepped card and filled with zero', filled, len(expected))

    # Predict across the ensemble and pull out the requested bout
    probabilities, predicted_classes = predict_xgboost_ensemble(ensemble, X, model_hyper_params = model_hyper_params)
    position = X.index.get_loc(bout_label)

    # Column 2 is fighter1 winning and column 0 is fighter2, matching Fighter1_win_prob and Fighter2_win_prob in evaluation.py.
    # The float() and int() calls matter because numpy's int64 is not a subclass of int and will not serialise to JSON
    return {
        'bout': fight.bout,
        'fighter1_odds': fight.fighter1_odds,
        'fighter2_odds': fight.fighter2_odds,
        'fighter1_true_closing_odds': row['fighter1_true_closing_odds'].iloc[0],
        'fighter1_win_prob': float(probabilities[position, 2]),
        'fighter2_win_prob': float(probabilities[position, 0]),
        'other_prob': float(probabilities[position, 1]),
        'predicted_class': int(predicted_classes[position])
    }

# Define the fights endpoint which gives a list of fights in the upcoming event that can be predicted.
# Not protected with authentication as nothing sensitive will be released here.
@app.get('/fights')
def list_fights(request: Request) -> list[FightSummary]:
    # ##
    # te = fp.ParquetFile('./data/final_dataset/final_dataset.parquet').to_pandas()
    # fights = te[te['outcome'].isna()][['bout', 'date', 'event', 'fighter1', 'fighter2']]
    # ##

    # Only the columns declared on FightSummary, since anything else would be dropped on the way out anyway
    fights = request.app.state.fights.copy()[['bout', 'date', 'event', 'fighter1', 'fighter2']]
    
    # records gives one dict per row, which FastAPI then validates against FightSummary
    return fights.to_dict(orient='records')
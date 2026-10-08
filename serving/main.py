from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
import fastparquet as fp
import joblib
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mma_functions.config import data_processing_config, model_hyper_params
from mma_functions.dataset import prep_data_for_model_training
from mma_functions.features import add_fighter_differentials
from mma_functions.modelling import predict_xgboost_ensemble
from mma_functions.odds import calculate_all_true_odds_columns


################################################################################
################################################################################

class FightRequest(BaseModel):

    model_config = ConfigDict(extra = 'forbid')
    # Pydantic will create the __init__ and attributes from these annotations as well as the validator , and OpenAPI schema
    bout: str
    fighter1_odds: float = Field(gt = 1.0)
    fighter2_odds: float = Field(gt = 1.0)

# Define the settings, which are used to configure the API such as model version etc
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = Path(__file__).parent / '.env')
    ensemble_path: str # These will come automatically from the environment ENSEMBLE_PATH, from an .env file like in docker or Kubernetes
    dataset_path: str

settings = Settings()

# Work out which odds column the model was trained on once here, rather than writing back to the config inside a request
if data_processing_config['impute_missing_betfair'] and model_hyper_params['odds_to_use'] == 'betfair_exch_odds':
    odds_to_use = 'betfair_exch_odds_imputed'
else:
    odds_to_use = model_hyper_params['odds_to_use']


################################################################################
################################################################################

# Define the startup actions in the lifespan. Use async as this is only run once and this is needed as FastAPI is an ASGI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the XGBoost ensemble
    app.state.model = joblib.load(settings.ensemble_path) # Can use settings object here without as an input because it is imported
    print('Loaded model ' + settings.ensemble_path)

    # Load the upcoming fights data
    full_data = fp.ParquetFile(settings.dataset_path).to_pandas()
    app.state.fights = full_data[full_data['outcome'].isna()].copy()
    print('Data loaded from ' + settings.dataset_path  + ', last modified: ' + str(Path(settings.dataset_path).stat().st_mtime))
    del full_data

    yield

app = FastAPI(lifespan = lifespan)


################################################################################
################################################################################

@app.get('/')
def placeholder():
    pass
# Emoty endpoint for now

@app.post('/predict')
def predict_fight(fight: FightRequest, request: Request):

    # Get the card data
    card = request.app.state.fights.copy()

    # Find the bout
    mask = card['bout'] == fight.bout

    # Raise exception if empty
    if not mask.any():
        raise HTTPException(status_code = 404, detail = f'No upcoming fight for bout {fight.bout}')
    elif mask.sum() > 1:
        raise HTTPException(status_code = 404, detail = f'Duplicate fight for bout {fight.bout} found in data')
    else:
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
        ensemble = request.app.state.model
        if any('_sub.att' in feature for feature in ensemble[0]['features']):
            X.columns = X.columns.str.replace('_sub_attempt', '_sub.att')

        # Force the exact columns and order the model expects, filling absent dummy levels with zero
        expected = ensemble[0]['model'].feature_names
        filled = len([col for col in expected if col not in X.columns])
        X = X.reindex(columns = expected, fill_value = 0)

        if filled > 0:
            print(str(filled) + ' of ' + str(len(expected)) + ' model features were missing from the prepped card and filled with zero')

        # Predict across the ensemble and pull out the requested bout
        probabilities, predicted_classes = predict_xgboost_ensemble(ensemble, X, model_hyper_params = model_hyper_params)
        position = X.index.get_loc(bout_label)

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


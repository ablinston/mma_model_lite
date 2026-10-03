from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()

class FightRequest(BaseModel):

    model_config = ConfigDict(extra = 'forbid')
    # Pydantic will create the __init__ and attributes from these annotations as well as the validator , and OpenAPI schema
    fight_id: str
    fighter1_odds: float = Field(gt = 1.0)
    fighter2_odds: float = Field(gt = 1.0)

@app.get('/')
def placeholder():
    pass
# Emoty endpoint for now

@app.post('/predict')
def predict_fight(fight: FightRequest):
    return {
        'fight_id': fight.fight_id,
        'fighter1_odds': fight.fighter1_odds,
        'fighter2_odds': fight.fighter2_odds,
        'fighter1_prob': 1 / fight.fighter1_odds
    }


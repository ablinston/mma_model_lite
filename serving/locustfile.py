import os
from locust import HttpUser, between, task

class PredictUser(HttpUser):

    # Pause between tasks, kept short
    wait_time = between(0.1, 0.5)

    # On startup, need to get the fights and a valid bout from them
    def on_start(self):

        # Get the fights and we want a bout from them
        fights = self.client.get('/fights').json()

        # Save a bout to self for the test
        self.bout = fights[0]['bout']

        # Set the authorisation token
        self.client.headers['Authorization'] = 'Bearer ' + os.environ['AUTH_TOKEN']

    @task
    def predict(self):

        # Request the prediction
        self.client.post(
            '/predict',
            json={'bout': self.bout,
                  'fighter1_odds': 1.8,
                  'fighter2_odds': 2.1
            })
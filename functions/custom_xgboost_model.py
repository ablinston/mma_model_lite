
class custom_xgboost_model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Loads the model from the artifact path when the model is loaded."""
        model_path = context.artifacts["model"]
        self.model = joblib.load(model_path)  # Load the model from a joblib file

    def predict(self, context, model_input):
        """Uses the custom prediction function."""
        # Call the function directly instead of a method on the model
        return predict_xgboost_ensemble(self.model, model_input)  # Pass the model and input to the function


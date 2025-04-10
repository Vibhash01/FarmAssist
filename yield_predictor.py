import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import os


class YieldPredictor:
    def __init__(self, data_path, model_path=None):
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}

        # Prepare the data
        self._encode_categorical_variables()

        # Define features and target variable
        X = self.data.drop("Yield (kg/ha)", axis=1)
        y = self.data["Yield (kg/ha)"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded Yield Predictor model from {model_path}")
        else:
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.mse = mean_squared_error(y_test, y_pred)
            self.r2 = r2_score(y_test, y_pred)
            self.accuracy_percentage = self.r2 * 100

            if model_path:
                self.save_model(model_path)
                print(f"Yield Predictor model saved to {model_path}")

    def _encode_categorical_variables(self):
        categorical_columns = ["Soil Type", "Planting Season", "Harvesting Season", "Subtype", "Crop Type", "Varieties"]
        for col in categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le

    def save_model(self, model_path):
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders
        }
        dump(model_data, model_path)

    def load_model(self, model_path):
        model_data = load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']

    def predict_yield(self, user_input):
        for col, le in self.label_encoders.items():
            if col in user_input:
                user_input[col] = le.transform([user_input[col]])[0]
            else:
                print(f"Missing value for {col}")
                return None

        user_df = pd.DataFrame([user_input])
        expected_cols = self.data.drop("Yield (kg/ha)", axis=1).columns

        missing_cols = set(expected_cols) - set(user_df.columns)
        if missing_cols:
            print(f"Missing columns in input: {missing_cols}")
            return None

        user_df = user_df[expected_cols]
        predicted_yield = self.model.predict(user_df)[0]
        return predicted_yield

    def predict_yield_cli(self, district, year, season, crop, area):
        user_input = {
            "Soil Type": "Loam",
            "Planting Season": season,
            "Harvesting Season": season,
            "Subtype": crop,
            "Crop Type": "Grain",
            "Varieties": "Basmati",
            "pH Level": 6.5,
            "Nitrogen Content (ppm)": 40,
            "Phosphorus Content (ppm)": 20,
            "Potassium Content (ppm)": 30,
            "Rainfall (mm)": 300,
            "Temperature (°C)": 25,
            "Humidity (%)": 60,
            "Sunlight Hours (per day)": 6,
            "Altitude (m)": 200,
            "Growing Period (days)": 90,
            "District": district,
            "Crop Year": year,
            "Area": area
        }

        input_features = self.data.drop("Yield (kg/ha)", axis=1).columns
        user_input_filtered = {k: user_input[k] for k in input_features if k in user_input}

        return self.predict_yield(user_input_filtered)

    def get_categorical_options(self):
        """Return available options for dropdowns in Streamlit"""
        options = {}
        for col in self.label_encoders:
            options[col] = list(self.label_encoders[col].classes_)
        return options

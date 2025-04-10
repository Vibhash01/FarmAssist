import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load
import os

class AgroFit:
    def __init__(self, data_path, model_path=None):
        # Load dataset
        self.data = pd.read_csv(data_path)

        # Encode categorical features
        self.soil_type_mapping = {soil: i for i, soil in enumerate(self.data['Soil Type'].unique())}
        self.planting_season_mapping = {season: i for i, season in enumerate(self.data['Planting Season'].unique())}
        self.harvesting_season_mapping = {season: i for i, season in enumerate(self.data['Harvesting Season'].unique())}

        self.data['Soil Type'] = self.data['Soil Type'].map(self.soil_type_mapping)
        self.data['Planting Season'] = self.data['Planting Season'].map(self.planting_season_mapping)
        self.data['Harvesting Season'] = self.data['Harvesting Season'].map(self.harvesting_season_mapping)

        # Select features
        self.feature_columns = ['Soil Type', 'pH Level', 'Nitrogen Content (ppm)', 'Phosphorus Content (ppm)',
                                'Potassium Content (ppm)', 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                                'Sunlight Hours (per day)', 'Altitude (m)', 'Planting Season', 'Harvesting Season',
                                'Growing Period (days)']
        self.features = self.data[self.feature_columns].copy()

        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

        # Apply PCA
        self.pca = PCA(n_components=8)
        self.pca_features = self.pca.fit_transform(self.scaled_features)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✅ Loaded KMeans model from {model_path}")
        else:
            self.kmeans = KMeans(n_clusters=100, random_state=0)
            self.kmeans.fit(self.pca_features)
            self.features['Cluster'] = self.kmeans.labels_
            self.data['Cluster'] = self.features['Cluster']
            self.silhouette_avg = silhouette_score(self.pca_features, self.kmeans.labels_)
            print(f"✅ Silhouette Score: {self.silhouette_avg:.3f}")

            if model_path:
                self.save_model(model_path)
                print(f"✅ KMeans model saved to {model_path}")

    def save_model(self, model_path):
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'soil_type_mapping': self.soil_type_mapping,
            'planting_season_mapping': self.planting_season_mapping,
            'harvesting_season_mapping': self.harvesting_season_mapping,
            'data': self.data
        }
        dump(model_data, model_path)

    def load_model(self, model_path):
        model_data = load(model_path)
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.soil_type_mapping = model_data['soil_type_mapping']
        self.planting_season_mapping = model_data['planting_season_mapping']
        self.harvesting_season_mapping = model_data['harvesting_season_mapping']
        self.data = model_data['data']

    def recommend_subtype(self, user_input):
        """
        Given input [N, P, K, temp, humidity, pH, rainfall], predict cluster and recommend most common crop subtype.
        """
        # Use mean values for missing columns
        default_values = self.data[self.feature_columns].mean()

        # Create full feature vector
        input_dict = {
            'Nitrogen Content (ppm)': user_input[0],
            'Phosphorus Content (ppm)': user_input[1],
            'Potassium Content (ppm)': user_input[2],
            'Temperature (°C)': user_input[3],
            'Humidity (%)': user_input[4],
            'pH Level': user_input[5],
            'Rainfall (mm)': user_input[6]
        }

        for col in self.feature_columns:
            if col not in input_dict:
                input_dict[col] = default_values[col]

        input_df = pd.DataFrame([input_dict])

        # Standardize and transform
        scaled = self.scaler.transform(input_df[self.feature_columns])
        reduced = self.pca.transform(scaled)

        # Predict cluster
        cluster_label = self.kmeans.predict(reduced)[0]

        # Recommend most frequent crop subtype in that cluster
        cluster_data = self.data[self.data['Cluster'] == cluster_label]
        if cluster_data.empty or 'Subtype' not in cluster_data:
            return "No recommendation found."

        recommended_subtype = cluster_data['Subtype'].mode()[0]
        return recommended_subtype

    def recommend_conditions(self, subtype, variety):
        crop_data = self.data[(self.data['Subtype'] == subtype) & (self.data['Varieties'] == variety)]
        if crop_data.empty:
            return "No data available for the specified subtype and variety."

        cluster_label = crop_data['Cluster'].mode()[0]
        cluster_data = self.data[self.data['Cluster'] == cluster_label]

        avg_conditions = cluster_data.mean(numeric_only=True)
        soil_type_mode = int(cluster_data['Soil Type'].mode()[0])
        planting_season_mode = int(cluster_data['Planting Season'].mode()[0])
        harvesting_season_mode = int(cluster_data['Harvesting Season'].mode()[0])

        soil_type = [k for k, v in self.soil_type_mapping.items() if v == soil_type_mode][0]
        planting_season = [k for k, v in self.planting_season_mapping.items() if v == planting_season_mode][0]
        harvesting_season = [k for k, v in self.harvesting_season_mapping.items() if v == harvesting_season_mode][0]

        recommended_conditions = avg_conditions.to_dict()
        recommended_conditions.update({
            'Soil Type': soil_type,
            'Planting Season': planting_season,
            'Harvesting Season': harvesting_season
        })

        return recommended_conditions

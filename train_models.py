import json
import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TRAFFIC_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Jam": 3}
WEATHER_MAP = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Storm": 3}
VEHICLE_MAP = {"Bicycle": 0, "Scooter": 1, "Car": 2}

FEATURE_NAMES = [
    "distance_km",
    "item_count",
    "base_prep_time",
    "traffic_code",
    "weather_code",
    "vehicle_code",
    "hour_of_day",
    "is_weekend"
]

def generate_synthetic_delivery_data(n_samples=6000, random_state=42):
    rng = np.random.RandomState(random_state)
    
    distance_km = rng.uniform(0.6, 12.0, size=n_samples)
    item_count = rng.randint(1, 10, size=n_samples)
    base_prep_time = rng.choice([8, 10, 12, 15, 18, 22, 25], size=n_samples)
    
    traffic_code = rng.choice([0, 1, 2, 3], p=[0.25, 0.45, 0.22, 0.08], size=n_samples)
    weather_code = rng.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.15, 0.05], size=n_samples)
    vehicle_code = rng.choice([0, 1, 2], p=[0.15, 0.70, 0.15], size=n_samples)
    
    hour_of_day = rng.randint(9, 24, size=n_samples)
    is_weekend = rng.choice([0, 1], p=[0.7, 0.3], size=n_samples)
    
    traffic_multipliers = np.array([1.0, 1.25, 1.6, 2.1])[traffic_code]
    weather_multipliers = np.array([1.0, 1.05, 1.35, 1.7])[weather_code]
    
    # Base speeds in km/h: bicycle=15, scooter=28, car=32
    vehicle_speeds = np.array([15.0, 28.0, 32.0])[vehicle_code]
    
    # Rush hour surge: 12-14 lunch, 19-21 dinner
    is_rush_hour = ((hour_of_day >= 12) & (hour_of_day <= 14)) | ((hour_of_day >= 19) & (hour_of_day <= 21))
    rush_factor = np.where(is_rush_hour, 1.15, 1.0)
    
    # Kitchen preparation time
    prep_actual = base_prep_time * (1.0 + 0.08 * (item_count - 1)) + rng.normal(0, 1.5, size=n_samples)
    prep_actual = np.clip(prep_actual, 5, 55)
    
    # Travel transit time in minutes
    effective_speed = vehicle_speeds / (traffic_multipliers * weather_multipliers * rush_factor)
    effective_speed = np.clip(effective_speed, 6.0, 45.0)
    transit_time = (distance_km / effective_speed) * 60.0
    
    # Rider dispatch / handoff overhead
    pickup_wait = 4.0 + rng.exponential(1.5, size=n_samples)
    
    # Total delivery ETA in minutes
    delivery_time = prep_actual + transit_time + pickup_wait + rng.normal(0, 1.0, size=n_samples)
    delivery_time = np.clip(delivery_time, 12.0, 95.0)
    
    X = np.column_stack([
        distance_km,
        item_count,
        base_prep_time,
        traffic_code,
        weather_code,
        vehicle_code,
        hour_of_day,
        is_weekend
    ])
    y = delivery_time
    
    return X, y

def train_and_save_models():
    os.makedirs("models", exist_ok=True)
    
    print("Generating 6,000 realistic delivery training instances...")
    X, y = generate_synthetic_delivery_data(n_samples=6000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Gradient Boosting Delivery ETA Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Results:")
    print(f"  R^2 Score: {r2:.4f} (Accuracy index)")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
    
    # Feature importances
    importances = model.feature_importances_
    feature_importance_list = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    ]
    
    model_path = os.path.join("models", "delivery_eta_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    meta_path = os.path.join("models", "model_metadata.json")
    metadata = {
        "model_name": "PandaBite AI Delivery ETA Predictor",
        "algorithm": "GradientBoostingRegressor (Ensemble Trees)",
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mae_minutes": round(float(mae), 2),
            "rmse_minutes": round(float(rmse), 2),
            "accuracy_percentage": round(float(r2 * 100), 2)
        },
        "features": FEATURE_NAMES,
        "feature_importances": feature_importance_list,
        "traffic_levels": list(TRAFFIC_MAP.keys()),
        "weather_conditions": list(WEATHER_MAP.keys()),
        "vehicle_types": list(VEHICLE_MAP.keys())
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    train_and_save_models()

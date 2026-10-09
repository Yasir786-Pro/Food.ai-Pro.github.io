# 🐼 foodpanda AI — Food Delivery Web App with Machine Learning

A complete, modern food delivery web application inspired by **Foodpanda**, featuring authentic pink branding, restaurant browsing, cart & checkout, live order tracking, and an integrated **Machine Learning Engine** with scikit-learn.

---

## 🚀 Key Features

### 1. 🍱 Foodpanda Experience
- **Authentic Branding**: Signature Foodpanda hot pink (`#D70F64`), sleek panda avatars, badge systems, modern typography.
- **Restaurant & Menu Browsing**: Multiple cuisines (Burgers, Woodfired Pizza, Asian & Dim Sum, Dum Biryani, Superfood Bowls, Boba & Desserts).
- **Cart & Vouchers**: Sliding cart drawer, dynamic item increments/decrements, voucher support (use promo code **`PANDA50`** for 50% OFF).
- **Checkout & Address Simulation**: Choose different delivery drop-offs or simulate delivery distances.

### 2. 🧠 Machine Learning Engine ("Build a Model in It")
- **Gradient Boosting ETA Regressor (`GradientBoostingRegressor`)**:
  - Predicts real-time delivery arrival minutes based on distance, traffic conditions, weather conditions, basket item count, and courier vehicle type.
  - **97.23% $R^2$ Score** with a Mean Absolute Error (MAE) of only **~2.2 minutes**.
  - **Live ML Sync**: Changing the weather or traffic in the header automatically recalculates delivery times across all restaurants in real time.
- **Smart Food Recommendation Model (Cosine Taste Matchmaker)**:
  - Vector similarity recommendation engine scoring dishes based on user's dietary preferences (veg/non-veg), spice tolerance (0 to 3), maximum budget, and mood (Comfort Food, Quick Bite, Guilt Free, Sweet Cravings).
  - Displays match scores (e.g. `98% Match`) and personalized explanation tags.

### 3. 🔬 Dedicated "AI & ML Lab" Dashboard
- **Interactive Live Simulator**: Real-time sliders for distance (0.5 to 15 km), items, kitchen prep time, weather, and traffic. Visualizes instant predictions, confidence intervals, and time allocation breakdown (Kitchen Prep vs. Transit vs. Handoff).
- **Model Explainability & Feature Importances**: Visualizes how much each feature influences delivery time.
- **Interactive Food Matchmaker**: Form to test recommendation results on the fly.

### 4. 🛵 Real-time Animated Delivery Tracker
- Interactive SVG map canvas showing kitchen pin, customer home pin, and live courier marker travelling along the delivery route.
- Step-by-step order progress timeline:
  1. Order Placed (Confirmed)
  2. Kitchen Cooking
  3. Rider Picked Up
  4. On the Way (Zooming)
  5. Delivered

---

## 🛠️ How to Run

### Step 1: Open Terminal / Command Prompt
Open PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd C:\Users\a\.gemini\antigravity\scratch\foodpanda_app
```

### Step 2: Ensure Dependencies Are Installed
The required dependencies (`flask`, `scikit-learn`, `numpy`, `joblib`) are already installed. If running on another machine:

```powershell
python -m pip install flask scikit-learn numpy joblib
```

### Step 3: Train the ML Model (Already Pre-trained!)
To re-train the model or regenerate synthetic training data:

```powershell
python train_models.py
```

### Step 4: Start the Web Application
Run the Flask server:

```powershell
python app.py
```

### Step 5: Open in Your Browser
Once the server starts, open your web browser and go to:

👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📂 Project Structure

```text
foodpanda_app/
│
├── app.py                      # Flask REST API & Web Server
├── train_models.py             # Scikit-learn ML training script (Gradient Boosting)
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and instructions
│
├── data/
│   └── restaurants.json        # Rich restaurant & dish catalog with pricing and tags
│
├── models/
│   ├── delivery_eta_model.joblib # Serialized GradientBoostingRegressor model
│   └── model_metadata.json     # R^2 score, MAE, RMSE & feature importances
│
├── static/
│   ├── css/
│   │   └── style.css           # Foodpanda styles, responsive grid & animations
│   └── js/
│       └── app.js              # Client state, cart, ML API calls, live tracking
│
└── templates/
    └── index.html              # Main single-page web app with AI Lab & Live Tracker
```

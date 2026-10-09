import json
import os
import time
import uuid
import datetime
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load restaurant database
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "restaurants.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "delivery_eta_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    RESTAURANTS = json.load(f)

# Load ML model and metadata
eta_model = None
model_metadata = {}

if os.path.exists(MODEL_PATH):
    try:
        eta_model = joblib.load(MODEL_PATH)
        print("Successfully loaded ML delivery ETA model.")
    except Exception as e:
        print(f"Error loading model: {e}")

if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)
    except Exception as e:
        print(f"Error loading model metadata: {e}")

TRAFFIC_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Jam": 3}
WEATHER_MAP = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Storm": 3}
VEHICLE_MAP = {"Bicycle": 0, "Scooter": 1, "Car": 2}

# In-memory orders store
ACTIVE_ORDERS = {}

def predict_single_eta(distance_km, item_count=2, base_prep_time=15,
                       traffic="Moderate", weather="Clear", vehicle="Scooter",
                       hour_of_day=None, is_weekend=0):
    if hour_of_day is None:
        hour_of_day = datetime.datetime.now().hour
    
    traffic_code = TRAFFIC_MAP.get(traffic, 1)
    weather_code = WEATHER_MAP.get(weather, 0)
    vehicle_code = VEHICLE_MAP.get(vehicle, 1)
    
    feature_row = np.array([[
        float(distance_km),
        float(item_count),
        float(base_prep_time),
        float(traffic_code),
        float(weather_code),
        float(vehicle_code),
        float(hour_of_day),
        float(is_weekend)
    ]])
    
    if eta_model is not None:
        prediction = float(eta_model.predict(feature_row)[0])
    else:
        # Fallback heuristic if model not loaded
        prediction = float(base_prep_time + (distance_km * 3.5) + (traffic_code * 4.0))
        
    return round(max(10.0, prediction), 1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/restaurants", methods=["GET"])
def get_restaurants():
    weather = request.args.get("weather", "Clear")
    traffic = request.args.get("traffic", "Moderate")
    vehicle = request.args.get("vehicle", "Scooter")
    search_query = request.args.get("search", "").strip().lower()
    category = request.args.get("category", "").strip().lower()
    
    results = []
    for r in RESTAURANTS:
        # Filtering
        if category and category != "all" and r["category"].lower() != category:
            continue
        
        if search_query:
            name_match = search_query in r["name"].lower()
            cuisine_match = search_query in r["cuisine"].lower()
            dish_match = any(search_query in d["name"].lower() for d in r["menu"])
            tag_match = any(search_query in t.lower() for t in r.get("tags", []))
            if not (name_match or cuisine_match or dish_match or tag_match):
                continue
                
        # ML predicted ETA calculation for each restaurant
        eta = predict_single_eta(
            distance_km=r["distanceKm"],
            item_count=2,
            base_prep_time=r["basePrepTime"],
            traffic=traffic,
            weather=weather,
            vehicle=vehicle
        )
        
        item_copy = dict(r)
        item_copy["predictedEta"] = int(round(eta))
        item_copy["etaRange"] = f"{int(round(eta - 2))}-{int(round(eta + 3))} mins"
        results.append(item_copy)
        
    return jsonify({
        "success": True,
        "count": len(results),
        "restaurants": results,
        "env_context": {
            "weather": weather,
            "traffic": traffic,
            "vehicle": vehicle
        }
    })

@app.route("/api/restaurants/<rest_id>", methods=["GET"])
def get_restaurant_detail(rest_id):
    for r in RESTAURANTS:
        if r["id"] == rest_id:
            eta = predict_single_eta(
                distance_km=r["distanceKm"],
                item_count=2,
                base_prep_time=r["basePrepTime"]
            )
            item_copy = dict(r)
            item_copy["predictedEta"] = int(round(eta))
            return jsonify({"success": True, "restaurant": item_copy})
    return jsonify({"success": False, "error": "Restaurant not found"}), 404

@app.route("/api/predict-eta", methods=["POST"])
def predict_eta():
    data = request.get_json() or {}
    try:
        distance_km = float(data.get("distance_km", 3.0))
        item_count = int(data.get("item_count", 2))
        base_prep_time = int(data.get("base_prep_time", 15))
        traffic = data.get("traffic", "Moderate")
        weather = data.get("weather", "Clear")
        vehicle = data.get("vehicle", "Scooter")
        hour = int(data.get("hour_of_day", datetime.datetime.now().hour))
        is_weekend = int(data.get("is_weekend", 0))
        
        predicted_minutes = predict_single_eta(
            distance_km=distance_km,
            item_count=item_count,
            base_prep_time=base_prep_time,
            traffic=traffic,
            weather=weather,
            vehicle=vehicle,
            hour_of_day=hour,
            is_weekend=is_weekend
        )
        
        # Component breakdown calculation for visual analytics
        prep_est = round(base_prep_time * (1.0 + 0.08 * (item_count - 1)), 1)
        speed = 28.0 if vehicle == "Scooter" else (15.0 if vehicle == "Bicycle" else 32.0)
        speed_modifier = [1.0, 1.25, 1.6, 2.1][TRAFFIC_MAP.get(traffic, 1)] * [1.0, 1.05, 1.35, 1.7][WEATHER_MAP.get(weather, 0)]
        transit_est = round((distance_km / (speed / speed_modifier)) * 60.0, 1)
        dispatch_est = round(predicted_minutes - prep_est - transit_est, 1)
        if dispatch_est < 2.0:
            dispatch_est = 3.5
            transit_est = round(predicted_minutes - prep_est - dispatch_est, 1)
            
        return jsonify({
            "success": True,
            "predicted_eta": round(predicted_minutes),
            "eta_range": f"{int(round(predicted_minutes - 2))}-{int(round(predicted_minutes + 3))} min",
            "confidence_score": 0.96,
            "breakdown": {
                "kitchen_prep_minutes": max(5, prep_est),
                "transit_minutes": max(3, transit_est),
                "rider_dispatch_minutes": max(2, dispatch_est)
            },
            "factors": {
                "traffic_impact": f"{traffic} traffic",
                "weather_impact": f"{weather} conditions",
                "vehicle": vehicle,
                "distance": f"{distance_km} km"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/recommend", methods=["POST"])
def recommend_dishes():
    data = request.get_json() or {}
    cuisine = data.get("cuisine", "All")
    max_price = float(data.get("max_price", 100))
    spicy_pref = data.get("spicy_pref", "Any")
    veg_only = bool(data.get("veg_only", False))
    mood = data.get("mood", "Any") # "Comfort Food", "Quick Bite", "Guilt Free", "Party", "Sweet Cravings"
    
    scored_dishes = []
    
    for r in RESTAURANTS:
        for d in r["menu"]:
            if veg_only and not d["isVeg"]:
                continue
            if d["price"] > max_price:
                continue
            
            score = 70.0 # base confidence
            reasons = []
            
            # Cuisine matching
            if cuisine != "All":
                if cuisine.lower() in r["cuisine"].lower() or cuisine.lower() in d["category"].lower():
                    score += 15.0
                    reasons.append(f"Favorite {cuisine} cuisine")
                else:
                    score -= 10.0
                    
            # Spicy matching
            if spicy_pref != "Any":
                target_spice = int(spicy_pref)
                diff = abs(d["spicyLevel"] - target_spice)
                if diff == 0:
                    score += 8.0
                    reasons.append("Exact spice match")
                elif diff == 1:
                    score += 3.0
                else:
                    score -= 8.0
                    
            # Mood heuristics
            if mood == "Quick Bite" and d["prepTimeMinutes"] <= 12:
                score += 7.0
                reasons.append("Fast 12-min prep time")
            elif mood == "Guilt Free" and (d["calories"] < 500 or d["isVeg"]):
                score += 8.0
                reasons.append(f"Low calorie ({d['calories']} kcal)")
            elif mood == "Comfort Food" and any(k in d["name"].lower() for k in ["burger", "pizza", "biryani", "cheese", "dumplings"]):
                score += 8.0
                reasons.append("Classic comfort feast")
            elif mood == "Sweet Cravings" and d["category"] in ["Desserts", "Beverages"]:
                score += 10.0
                reasons.append("Satisfies sweet cravings")
                
            # Popularity / restaurant rating bonus
            score += (r["rating"] - 4.0) * 8.0
            
            final_match = min(99, max(65, int(round(score))))
            
            if not reasons:
                reasons.append("Popular customer choice")
                
            dish_item = dict(d)
            dish_item["restaurantId"] = r["id"]
            dish_item["restaurantName"] = r["name"]
            dish_item["restaurantImage"] = r["image"]
            dish_item["distanceKm"] = r["distanceKm"]
            dish_item["matchScore"] = final_match
            dish_item["matchReason"] = " • ".join(reasons[:2])
            scored_dishes.append(dish_item)
            
    scored_dishes.sort(key=lambda x: x["matchScore"], reverse=True)
    return jsonify({
        "success": True,
        "recommendations": scored_dishes[:8]
    })

@app.route("/api/order", methods=["POST"])
def place_order():
    data = request.get_json() or {}
    items = data.get("items", [])
    if not items:
        return jsonify({"success": False, "error": "Order cannot be empty"}), 400
        
    order_id = "FP-" + uuid.uuid4().hex[:8].upper()
    total_items = sum(item.get("quantity", 1) for item in items)
    
    # Calculate delivery ETA using ML model
    # Use restaurant distance if available
    first_item_rest_id = items[0].get("restaurantId", "rest-1")
    restaurant = next((r for r in RESTAURANTS if r["id"] == first_item_rest_id), RESTAURANTS[0])
    
    predicted_eta = predict_single_eta(
        distance_km=restaurant["distanceKm"],
        item_count=total_items,
        base_prep_time=restaurant["basePrepTime"]
    )
    
    now = datetime.datetime.now()
    estimated_arrival = (now + datetime.timedelta(minutes=predicted_eta)).strftime("%I:%M %p")
    
    order_record = {
        "orderId": order_id,
        "createdAt": now.isoformat(),
        "createdTimestamp": time.time(),
        "restaurantName": restaurant["name"],
        "restaurantImage": restaurant["image"],
        "items": items,
        "totalAmount": data.get("totalAmount", 24.50),
        "deliveryAddress": data.get("deliveryAddress", "124 Cherry Blossom Lane, Apt 4B"),
        "predictedEtaMinutes": int(round(predicted_eta)),
        "estimatedArrival": estimated_arrival,
        "rider": {
            "name": "Panda Rider Alex",
            "vehicle": "Yamaha NMAX (Motorbike)",
            "rating": 4.95,
            "deliveries": 1840,
            "avatar": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80"
        }
    }
    
    ACTIVE_ORDERS[order_id] = order_record
    
    return jsonify({
        "success": True,
        "order": order_record
    })

@app.route("/api/order/<order_id>/track", methods=["GET"])
def track_order(order_id):
    order = ACTIVE_ORDERS.get(order_id)
    if not order:
        # Default mock order for immediate demonstration if user tests tracking directly
        order = {
            "orderId": order_id,
            "createdTimestamp": time.time() - 40,
            "restaurantName": "The Gourmet Burger Republic",
            "restaurantImage": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?auto=format&fit=crop&w=700&q=80",
            "predictedEtaMinutes": 24,
            "deliveryAddress": "124 Cherry Blossom Lane, Apt 4B",
            "rider": {
                "name": "Panda Rider Alex",
                "vehicle": "Yamaha NMAX (Motorbike)",
                "rating": 4.95,
                "deliveries": 1840,
                "avatar": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80"
            }
        }
        
    elapsed_seconds = time.time() - order.get("createdTimestamp", time.time())
    
    # Progress simulation:
    # 0 - 15s: Order Confirmed
    # 15 - 35s: Preparing in Kitchen
    # 35 - 55s: Rider Picked Up
    # 55 - 85s: Rider on the way (live road progress)
    # > 85s: Arrived & Delivered
    if elapsed_seconds < 15:
        step_idx = 0
        status_text = "Order Confirmed by Restaurant"
        rider_progress_percent = 5
    elif elapsed_seconds < 35:
        step_idx = 1
        status_text = "Kitchen is cooking your meal with care"
        rider_progress_percent = 25
    elif elapsed_seconds < 55:
        step_idx = 2
        status_text = "Rider picked up order from kitchen"
        rider_progress_percent = 50
    elif elapsed_seconds < 90:
        step_idx = 3
        # gradual progress from 50 to 95%
        p = 50 + int((elapsed_seconds - 55) / 35 * 45)
        rider_progress_percent = min(95, p)
        status_text = f"Rider is zooming towards your location ({order.get('predictedEtaMinutes', 20) - int(elapsed_seconds//15)} mins left)"
    else:
        step_idx = 4
        status_text = "Order Delivered! Enjoy your meal!"
        rider_progress_percent = 100
        
    return jsonify({
        "success": True,
        "orderId": order_id,
        "stepIndex": step_idx,
        "statusText": status_text,
        "progressPercent": rider_progress_percent,
        "remainingMinutes": max(1, order.get("predictedEtaMinutes", 22) - int(elapsed_seconds // 10)),
        "restaurantName": order.get("restaurantName"),
        "deliveryAddress": order.get("deliveryAddress"),
        "rider": order.get("rider")
    })

@app.route("/api/model-stats", methods=["GET"])
def get_model_stats():
    return jsonify({
        "success": True,
        "metadata": model_metadata
    })

if __name__ == "__main__":
    print("Starting PandaBite Food Delivery Flask App on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)

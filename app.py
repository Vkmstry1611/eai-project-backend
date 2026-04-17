from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend requests


@app.route('/')
def home():
    return "Smart Water Monitoring Backend Running 🚀"


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json

        channel_id = data.get('channel_id')
        api_key = data.get('api_key')
        tank_height = float(data.get('tank_height'))

        if not channel_id or not api_key:
            return jsonify({"error": "Missing channel_id or api_key"}), 400

        # 📡 Fetch data from ThingSpeak
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results=50"
        response = requests.get(url)
        feeds = response.json().get('feeds', [])

        time = []
        levels = []
        distances = []

        for i, entry in enumerate(feeds):
            if entry.get('field1') is not None:
                water_level = float(entry['field1'])
                distance = float(entry['field2']) if entry.get('field2') is not None else (tank_height - water_level)

                water_level = max(0, min(water_level, tank_height))

                time.append(i)
                levels.append(water_level)
                distances.append(distance)

        if len(levels) < 3:
            return jsonify({"error": "Not enough data for prediction"}), 400

        # 🧠 Linear Regression using NumPy (NO sklearn)
        X = np.array(time)
        y = np.array(levels)

        # Fit line: y = mx + c
        m, c = np.polyfit(X, y, 1)

        # 🔮 Predict next 10 values
        future = np.arange(len(time), len(time) + 10)
        predictions = (m * future + c).tolist()

        # 🚨 Leakage Detection
        leakage = False
        if levels[-1] < levels[-2] - 5:
            leakage = True

        # 📈 Trend detection
        trend = "stable"
        if predictions[-1] < levels[-1]:
            trend = "decreasing"
        elif predictions[-1] > levels[-1]:
            trend = "increasing"

        # ⏱️ Time to empty (in hours)
        time_to_empty = None
        if m < 0:  # level is dropping
            steps_to_empty = -levels[-1] / m
            # each reading is every 15 seconds
            time_to_empty = round((steps_to_empty * 15) / 3600, 2)  # hours

        # 🔍 Anomaly detection: actual vs predicted residuals
        fitted = (m * X + c).tolist()
        residuals = [abs(levels[i] - fitted[i]) for i in range(len(levels))]
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        threshold = mean_res + 2 * std_res
        anomalies = [i for i, r in enumerate(residuals) if r > threshold]

        return jsonify({
            "levels": levels,
            "predictions": predictions,
            "distances": distances,
            "current_level": levels[-1],
            "leakage": leakage,
            "trend": trend,
            "time_to_empty": time_to_empty,
            "anomalies": anomalies,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
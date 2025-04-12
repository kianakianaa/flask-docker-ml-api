from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/predict")
def predict():
    x = float(request.args.get("x", 0))
    w = float(request.args.get("w", 0))
    y_pred = model.predict(w, x)
    
    # Log prediction
    with open("output.txt", "w") as f:
        f.write(f"Input x: {x}\nPrediction: {y_pred}\n")
    
    return jsonify({"x": x, "w": w, "prediction": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

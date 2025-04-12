from flask import Flask, request, jsonify
import statsmodels.api as sm
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/predict")
def predict():
    x = float(request.args.get("x", 0))
    w = float(request.args.get("w", 0))
    new_data = pd.DataFrame({'const': [1], 'w': [w], 'x': [x]})
    y_pred = model.predict(new_data)
    
    # Log prediction
    with open("output.txt", "w") as f:
        f.write(f"Input x: {x}\nPrediction: {y_pred}\n")
    
    return jsonify({"x": x, "w": w, "prediction": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

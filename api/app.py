"""
Fraud Detection API - Flask Application
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import complete pipeline (not just model_loader)
from api.complete_pipeline import pipeline

app = Flask(__name__)
CORS(app)

# ============================================
# HEALTH CHECK
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': pipeline.model is not None
    })

# ============================================
# HELLO WORLD
# ============================================
@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({
        'message': 'Fraud Detection API is running!',
        'endpoints': {
            '/': 'GET - Hello World',
            '/health': 'GET - Health check',
            '/predict': 'POST - Fraud prediction'
        },
        'version': '1.0.0'
    })

# ============================================
# PREDICTION
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction using complete pipeline
        prediction, probability = pipeline.predict(input_df)
        
        response = {
            'prediction': int(prediction[0]),
            'fraud_probability': float(probability[0]),
            'fraud_label': 'FRAUD' if prediction[0] == 1 else 'NON-FRAUD',
            'confidence': float(max(probability[0], 1 - probability[0])),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# MODEL INFO
# ============================================
@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify(pipeline.get_info())

# ============================================
# RUN
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 FRAUD DETECTION API")
    print("=" * 60)
    print(f"📂 Model loaded: {pipeline.model is not None}")
    print(f"🔧 Model type: {type(pipeline.model).__name__ if pipeline.model else 'None'}")
    print("=" * 60)
    print("📍 Endpoints:")
    print("   GET  /           - Hello World")
    print("   GET  /health     - Health Check")
    print("   POST /predict    - Single Prediction")
    print("   GET  /model/info - Model Info")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
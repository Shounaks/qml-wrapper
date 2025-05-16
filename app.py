from flask import Flask, request, jsonify
import joblib
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

# Initialize Flask app
app = Flask(__name__)

# Load the scaler and VQC components
try:
    vqc: VQC = VQC.load('vqc_model.dill')
    scaler = joblib.load('scaler.pkl')
    print("Scaler and VQC components loaded successfully")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'scaler.pkl' and 'vqc_components.pkl' are in the same directory")
    exit(1)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features in the correct order
        features = [
            data['flow_duration'],
            data['total_fwd_packets'],
            data['total_backward_packets'],
            data['fwd_packet_length_max']
        ]

        # Convert to numpy array and reshape for scaler
        features = np.array(features).reshape(1, -1)
        print(f"Input features shape: {features.shape}")  # Debug

        # Validate feature shape
        if features.shape != (1, 4):
            return jsonify({
                'error': f'Expected feature shape (1, 4), got {features.shape}',
                'status': 'error'
            }), 400

        # Normalize features
        features_scaled = scaler.transform(features)
        print(f"Scaled features shape: {features_scaled.shape}")  # Debug

        # Make prediction
        prediction = vqc.predict(features_scaled)
        print(f"Raw prediction: {prediction}, shape: {np.shape(prediction)}")  # Debug

        # Handle scalar or array output
        if np.isscalar(prediction) or prediction.ndim == 0:
            prediction_value = int(prediction.item())
        else:
            prediction_value = int(prediction[0])

        # Map prediction to label
        label = 'Brute Force' if prediction_value == 1 else 'BENIGN'

        # Return JSON response
        return jsonify({
            'prediction': label,
            'status': 'success'
        })

    except KeyError as e:
        return jsonify({
            'error': f'Missing feature: {str(e)}',
            'status': 'error'
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


# Health check endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'VQC Prediction API is running',
        'status': 'success'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

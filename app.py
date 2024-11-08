from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the model, scaler, and label encoder at startup
MODEL_PATH = 'model/ransomware_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Function to load and preprocess individual CSV files
def load_and_preprocess_csvs(file_access, network_traffic, system_performance, user_behavior):
    # Load dataframes
    file_access_df = pd.read_csv(file_access)
    network_traffic_df = pd.read_csv(network_traffic)
    system_performance_df = pd.read_csv(system_performance)
    user_behavior_df = pd.read_csv(user_behavior)
    
    # Ensure Timestamp column is present in each CSV
    for df, name in zip([file_access_df, network_traffic_df, system_performance_df, user_behavior_df], 
                        ['file_access', 'network_traffic', 'system_performance', 'user_behavior']):
        if 'Timestamp' not in df.columns:
            raise KeyError(f"'Timestamp' column missing in {name}. Please check the CSV file.")
    
    # Merge datasets on the 'Timestamp' column
    df = pd.merge(file_access_df, network_traffic_df, on='Timestamp', how='outer')
    df = pd.merge(df, system_performance_df, on='Timestamp', how='outer')
    df = pd.merge(df, user_behavior_df, on='Timestamp', how='outer')
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Encode categorical variables
    if 'File_Operation' in df.columns:
        df['File_Operation'] = label_encoder.transform(df['File_Operation'].fillna('unknown'))
    if 'User' in df.columns:
        df['User'] = label_encoder.transform(df['User'].fillna('unknown'))
    
    return df

@app.route('/predict', methods=['POST'])
def predict_ransomware():
    # Check if all required files are in the request
    required_files = ['file_access', 'network_traffic', 'system_performance', 'user_behavior']
    if not all(file in request.files for file in required_files):
        return jsonify({'error': 'All files (file_access, network_traffic, system_performance, user_behavior) are required.'}), 400

    # Read files from the request
    files = {name: request.files[name] for name in required_files}

    # Load and preprocess data
    try:
        df = load_and_preprocess_csvs(files['file_access'], files['network_traffic'], files['system_performance'], files['user_behavior'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Define feature columns (ensure these match the model's features)
    feature_columns = ['File_Operation', 'File_Size_MB', 'Network_Bytes', 'CPU_Usage', 'Disk_IO', 'Memory_Usage', 'Login_Success', 'Privilege_Escalation']
    X_new = df[feature_columns]

    # Scale the features
    X_new_scaled = scaler.transform(X_new)

    # Predict ransomware
    df['Ransomware_Prediction'] = model.predict(X_new_scaled)
    ransomware_events = df[df['Ransomware_Prediction'] == 1]

    # Convert results to JSON
    predictions = ransomware_events[['Timestamp', 'Ransomware_Prediction']].to_dict(orient='records')
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
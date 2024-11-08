from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import plotly.graph_objects as go
import plotly.express as px
import os

main_bp = Blueprint('main', __name__)

# Helper function to get model path
def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), 'models', filename)

def load_model_and_scaler():
    model_path = get_model_path('ransomware_model.pkl')
    scaler_path = get_model_path('scaler.pkl')
    label_encoder_path = get_model_path('label_encoder.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, scaler, label_encoder

def process_dataframe(df):
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    if 'File_Operation' in df.columns:
        df['File_Operation'] = label_encoder.fit_transform(df['File_Operation'])
    if 'User' in df.columns:
        df['User'] = label_encoder.fit_transform(df['User'])
    
    return df, label_encoder

@main_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Load files from request
        files = request.files
        
        # Read CSV files
        file_access = pd.read_csv(files['file_access'])
        network_traffic = pd.read_csv(files['network_traffic'])
        system_performance = pd.read_csv(files['system_performance'])
        user_behavior = pd.read_csv(files['user_behavior'])
        
        # Merge datasets
        df = pd.merge(file_access, network_traffic, on='Timestamp', how='outer')
        df = pd.merge(df, system_performance, on='Timestamp', how='outer')
        df = pd.merge(df, user_behavior, on='Timestamp', how='outer')
        
        # Process dataframe
        df, label_encoder = process_dataframe(df)
        
        # Load model and make predictions
        model, scaler, saved_label_encoder = load_model_and_scaler()
        
        feature_columns = ['File_Operation', 'File_Size_MB', 'Network_Bytes', 
                          'CPU_Usage', 'Disk_IO', 'Memory_Usage', 'Login_Success', 
                          'Privilege_Escalation']
        
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Prepare response
        df['Ransomware_Prediction'] = predictions
        ransomware_events = df[df['Ransomware_Prediction'] == 1]
        
        response = {
            'status': 'success',
            'total_predictions': len(predictions),
            'ransomware_detected': len(ransomware_events),
            'timestamps_affected': ransomware_events['Timestamp'].tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@main_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200
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
    model_path = get_model_path('ransomware_model_3.0.pkl')
    scaler_path = get_model_path('scaler_3.0.pkl')
    label_encoder_path = get_model_path('label_encoder_3.0.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, scaler, label_encoder

def process_dataframe(df, label_encoder):
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Encode categorical variables using the loaded label encoder
    if 'File_Operation' in df.columns:
        df['File_Operation'] = df['File_Operation'].apply(lambda x: label_encoder.transform([x])[0]
                                                          if x in label_encoder.classes_ 
                                                          else -1)
    if 'User' in df.columns:
        df['User'] = df['User'].apply(lambda x: label_encoder.transform([x])[0]
                                      if x in label_encoder.classes_
                                      else -1)
    
    return df

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
        
        # Debugging unique types for each column
        debug_info = {}
        for column in df.columns:
            unique_types = df[column].apply(type).unique()
            debug_info[column] = str(unique_types)
        
        # Load model and encoders
        model, scaler, label_encoder = load_model_and_scaler()
        
        # Process dataframe
        df = process_dataframe(df, label_encoder)
        
        # Define feature columns and ensure they are present
        feature_columns = [
            'File_Operation', 'File_Size_MB', 'Network_Bytes', 
            'CPU_Usage', 'Disk_IO', 'Memory_Usage', 'Login_Success', 
            'Privilege_Escalation'
        ]
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise KeyError(f"Missing required feature columns: {missing_features}")
        
        # Extract features and scale them
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        
        # Make predictions
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
        # Return error with debug info
        return jsonify({
            'status': 'error',
            'message': str(e),
            'debug': debug_info
        }), 400

@main_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200
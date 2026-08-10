
import joblib
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FraudPipeline:
    """Complete pipeline from raw input to prediction"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FraudPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Store all artifacts
        self.model = None
        self.preprocessor = None
        self.selector = None
        self.feature_names = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all artifacts"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            
            print(f"📂 Loading artifacts from: {models_dir}")
            
            self.model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
            self.preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
            self.selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))
            
            # Load feature info
            try:
                feature_info = joblib.load(os.path.join(models_dir, 'feature_info.pkl'))
                self.feature_names = feature_info.get('all_features', [])
            except:
                self.feature_names = []
            
            print(f"✅ Artifacts loaded successfully!")
            print(f"   - Model: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {str(e)}")
            raise e
    
    def engineer_features(self, df):
        """Apply feature engineering (same as training)"""
        df_fe = df.copy()
        
        # Handle timestamp
        if 'timestamp' in df_fe.columns:
            df_fe['timestamp'] = pd.to_datetime(df_fe['timestamp'])
            df_fe['hour'] = df_fe['timestamp'].dt.hour
            df_fe['day_of_week'] = df_fe['timestamp'].dt.dayofweek
            df_fe['day_of_month'] = df_fe['timestamp'].dt.day
            df_fe['month'] = df_fe['timestamp'].dt.month
        else:
            # Use defaults if timestamp not provided
            df_fe['hour'] = 12
            df_fe['day_of_week'] = 0
            df_fe['day_of_month'] = 15
            df_fe['month'] = 8
        
        df_fe['is_weekend'] = df_fe['day_of_week'].isin([5, 6]).astype(int)
        df_fe['is_night'] = ((df_fe['hour'] >= 22) | (df_fe['hour'] <= 5)).astype(int)
        
        # Ratio features
        df_fe['amount_vs_customer_avg'] = df_fe['amount_to_avg_ratio']
        df_fe['amount_vs_merchant_avg'] = df_fe['transaction_amount'] / df_fe['avg_transaction_amount']
        df_fe['amount_vs_merchant_avg'] = df_fe['amount_vs_merchant_avg'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Customer features
        df_fe['customer_age_bracket'] = pd.cut(df_fe['age'], 
                                               bins=[0, 25, 40, 60, 100], 
                                               labels=['Young', 'Adult', 'Middle_Aged', 'Senior'])
        
        df_fe['account_age_years'] = df_fe['account_age_months'] / 12
        df_fe['is_new_customer'] = (df_fe['account_age_months'] < 6).astype(int)
        df_fe['credit_score_bracket'] = pd.cut(df_fe['credit_score'], 
                                               bins=[0, 580, 670, 740, 800, 850], 
                                               labels=['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent'])
        
        # Merchant features
        df_fe['merchant_age_years'] = 2026 - df_fe['established_year']
        df_fe['is_old_merchant'] = (df_fe['merchant_age_years'] > 20).astype(int)
        df_fe['merchant_fraud_rate'] = df_fe['fraud_reports'] / (df_fe['merchant_age_years'] + 1)
        
        # Interaction features
        df_fe['high_risk_customer_high_risk_merchant'] = ((df_fe['risk_category'] == 'High') & 
                                                           df_fe['is_high_risk']).astype(int)
        df_fe['high_amount_high_risk'] = ((df_fe['transaction_amount'] > df_fe['transaction_amount'].median()) & 
                                           df_fe['is_high_risk']).astype(int)
        df_fe['international_online'] = (df_fe['is_international'] & df_fe['is_online']).astype(int)
        
        # Velocity
        df_fe['time_since_last_txn_hours'] = df_fe.get('time_since_last_txn_hours', 24.0)
        
        return df_fe
    
    def predict(self, raw_data):
        """
        End-to-end prediction from raw data
        raw_data: pandas DataFrame
        returns: prediction and probability
        """
        # Step 1: Feature engineering
        engineered = self.engineer_features(raw_data)
        
        # Step 2: Preprocess
        processed = self.preprocessor.transform(engineered)
        
        # Step 3: Feature selection
        selected = self.selector.transform(processed)
        
        # Step 4: Predict
        prediction = self.model.predict(selected)
        probability = self.model.predict_proba(selected)[:, 1]
        
        return prediction, probability
    
    def get_info(self):
        return {
            'model_type': type(self.model).__name__,
            'is_loaded': self.model is not None
        }

# Global instance
pipeline = FraudPipeline()
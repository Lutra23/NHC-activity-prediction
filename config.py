import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # 添加其他配置项
    BOND_ORDER_MODEL = os.path.join(MODEL_PATH, 'bond_order_model.joblib')
    AROMATICITY_MODEL = os.path.join(MODEL_PATH, 'aromaticity_model.joblib') 
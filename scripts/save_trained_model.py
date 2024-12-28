import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import joblib
import pandas as pd
from app.config import Config

def save_model():
    """
    保存训练好的模型到配置指定的位置
    """
    try:
        # 创建模型目录
        Config.MODEL_DIR.mkdir(exist_ok=True)
        
        # 使用项目根目录构建模型文件的完整路径
        model_path = project_root / 'model' / 'rf_bond_order_predictor.joblib'
        
        # 检查源模型文件是否存在
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        # 从训练笔记本加载模型数据
        print(f"正在从 {model_path} 加载模型...")
        model_data = joblib.load(model_path)
        
        # 保存到应用的模型目录
        joblib.dump(model_data, Config.BOND_ORDER_MODEL)
        print(f"模型已成功保存到: {Config.BOND_ORDER_MODEL}")
        
        # 打印模型信息
        print("\n模型信息:")
        print(f"特征数量: {len(model_data['feature_names'])}")
        print(f"模型类型: {type(model_data['model']).__name__}")
        
        if hasattr(model_data['model'], 'feature_importances_'):
            # 获取前10个最重要的特征
            importance = pd.DataFrame({
                'feature': model_data['feature_names'],
                'importance': model_data['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n前10个最重要的特征:")
            print(importance.head(10))
            
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    save_model() 
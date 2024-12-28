import os
from pathlib import Path
from rdkit import Chem
from app.models.bond_order import init_model as init_bond_order_model
from app.models.bond_order import predict_bond_order
from app.models.geometry_optimizer import GeometryOptimizer

def setup_directories():
    """创建必要的目录"""
    dirs = ['xyz-carbene', 'xyz-pi', 'test_xyz']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("目录创建完成")

def test_prediction():
    """测试预测功能"""
    # 创建目录
    setup_directories()
    
    # 初始化模型
    try:
        init_bond_order_model("xyz-carbene")
        print("模型初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return
    
    # 测试分子 - 一个简单的卡宾结构
    smiles = "CN(C)[C]"  # N,N-二甲基卡宾
    carbene_idx = 3      # 卡宾碳的索引
    hash_value = "test_carbene_1"
    
    try:
        # 进行预测
        result = predict_bond_order(smiles, carbene_idx, hash_value)
        
        # 打印结果
        print("\n预测结果:")
        print(f"SMILES: {smiles}")
        print(f"卡宾位置: {carbene_idx}")
        print(f"键级: {result['bond_orders']}")
        print(f"置信度: {result['confidence']}")
        
    except Exception as e:
        print(f"预测失败: {str(e)}")

if __name__ == "__main__":
    test_prediction() 
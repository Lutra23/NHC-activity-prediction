import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from app.models.bond_order import BondOrderModel
from app.utils.errors import ModelError, MoleculeError, CarbeneError

class TestBondOrder:
    @pytest.fixture
    def model(self):
        """创建模型实例"""
        return BondOrderModel()
        
    def test_basic_prediction(self, model):
        """测试基本的键级预测功能"""
        # 使用二甲基卡宾: CC(C)[C]
        result = model.predict("[H]c1c([H])c([H])c(-n2c3c([H])c([H])c([H])c([H])c3c3c([H])c(-[n+]4cn5c(=O)c6c([H])c([H])c([H])c([H])c6c6c([H])c([H])c([H])c4c65)c([H])c([H])c32)c([H])c1[H]", carbene_idx=16)
        assert isinstance(result, dict)
        assert 'bond_orders' in result
        assert 'confidence' in result
        assert isinstance(result['bond_orders'], dict)
        assert len(result['bond_orders']) > 0
        assert all(isinstance(v, float) for v in result['bond_orders'].values())
        assert 0 <= result['confidence'] <= 1
        
    def test_cache_mechanism(self, model):
        """测试缓存机制"""
        smiles = "[H]c1c([N+](=O)[O-])c([H])c2c(c1[H])n1c(N3C([H])([H])C([H])([H])OC([H])([H])C3([H])[H])[n+]3c([H])c([H])c([H])c([H])c3c1c1c3c([H])c([H])c([H])c([H])[n+]3cn21"
        result1 = model.predict(smiles, carbene_idx=11)
        result2 = model.predict(smiles, carbene_idx=11)
        assert result1['bond_orders'] == result2['bond_orders']
        
    def test_batch_prediction(self, model):
        """测试批量预测"""
        molecules = [
            ("[H]c1c([N+](=O)[O-])c([H])c2c(c1[H])n1c(N3C([H])([H])C([H])([H])OC([H])([H])C3([H])[H])[n+]3c([H])c([H])c([H])c([H])c3c1c1c3c([H])c([H])c([H])c([H])[n+]3cn21", 11),
            ("[H]c1c([H])c([H])c2c(c1[H])-c1c([H])c([H])c([H])c([H])[n+]1C2([H])c1c2c(c([H])c3c([H])c([H])c([H])c([H])c13)Sc1c([H])c([H])c([H])c3c1[n+]-2cn3-c1c([H])c(C([H])([H])[H])c([H])c(C([H])([H])[H])c1[H]", 9),
            ("[H]OOOSC#Cn1c[n+](C([H])([H])[H])c2c([H])c(Cl)c(Cl)c([H])c21", 3)
        ]
        results = model.predict_batch(molecules)
        assert len(results) == 3
        assert all(isinstance(r, dict) or r is None for r in results)
        
    def test_invalid_inputs(self, model):
        """测试无效输入"""
        with pytest.raises(ModelError):
            model.predict("invalid", carbene_idx=0)
            
        with pytest.raises(ModelError):
            model.predict("[H]OOOSC#Cn1c[n+](C([H])([H])[H])c2c([H])c(Cl)c(Cl)c([H])c21", carbene_idx=60)  # 超出范围的卡宾位置
            
    def test_edge_cases(self, model):
        """测试边界情况"""
        # 最简单的有效卡宾结构
        result = model.predict("[H]OOOSC#Cn1c[n+](C([H])([H])[H])c2c([H])c(Cl)c(Cl)c([H])c21", carbene_idx=3)
        assert isinstance(result, dict)
        
        # 带有更多取代基的卡宾
        result = model.predict("[H]OOOSC#Cn1c[n+](C([H])([H])[H])c2c([H])c(Cl)c(Cl)c([H])c21", carbene_idx=3)
        assert isinstance(result, dict)
        
    def test_prediction_consistency(self, model):
        """测试预测一致性"""
        # 使用对称的二甲基卡宾
        smiles = "[H]OOOSC#Cn1c[n+](C([H])([H])[H])c2c([H])c(Cl)c(Cl)c([H])c21"
        result1 = model.predict(smiles, carbene_idx=3)
        result2 = model.predict(smiles, carbene_idx=3)
        assert abs(list(result1['bond_orders'].values())[0] - 
                  list(result2['bond_orders'].values())[0]) < 0.1
                  
    def test_error_handling_in_batch(self, model):
        """测试批量预测中的错误处理"""
        molecules = [
            ("CC(C)[C]", 3),    # 有效
            ("invalid", 0),      # 无效SMILES
            ("CCC", 6),         # 无效位置
            ("CC(C)(C)[C]", 4)  # 有效
        ]
        results = model.predict_batch(molecules)
        assert len(results) == 4
        assert isinstance(results[0], dict)  # 第一个应该成功
        assert results[1] is None  # 第二个应该失败
        assert results[2] is None  # 第三个应该失败
        assert isinstance(results[3], dict)  # 第四个应该成功
        
    def test_model_features(self, model):
        """测试特征提取"""
        mol = Chem.MolFromSmiles("CC(C)[C]")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        features = model.extract_features(mol, carbene_idx=3)
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 269  # 验证特征维度
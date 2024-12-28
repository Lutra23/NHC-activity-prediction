import unittest
import numpy as np
from rdkit import Chem
from app.models.aromaticity import predict_aromaticity, init_model
from app.utils.errors import ModelError, MoleculeError
import pytest
from app.models.aromaticity import AromaticityPredictor
from app.utils.errors import MoleculeError, ModelError


class TestPiBondOrder:
    @pytest.fixture
    def predictor(self):
        return AromaticityPredictor()
        
    def test_benzene_pi_bond_order(self, predictor):
        """测试苯环的π键级"""
        result = predictor.predict("c1ccccc1")
        assert 'pi_bond_order' in result
        assert 0.6 <= result['pi_bond_order'] <= 0.65
        
    def test_cyclopentadiene_pi_bond_order(self, predictor):
        """测试环戊二烯的π键级"""
        result = predictor.predict("C1=CC=CC1")
        assert 'pi_bond_order' in result
        assert 0.6 <= result['pi_bond_order'] <= 0.7 # 环戊二烯的π键级应该接近2.0
        
    def test_cyclohexane_pi_bond_order(self, predictor):
        """测试环己烷的π键级"""
        result = predictor.predict("C1CCCCC1")
        assert 'pi_bond_order' in result
        assert 0.0 <= result['pi_bond_order'] <= 0.2  # 环己烷的π键级应该接近0
        
    def test_complex_molecule_pi_bond_order(self, predictor):
        """测试复杂分子的π键级"""
        # 萘
        result = predictor.predict("c1ccc2ccccc2c1")
        assert 'pi_bond_order' in result
        assert result['pi_bond_order'] > 0
        
        # 吡啶
        result = predictor.predict("c1ccncc1")
        assert 'pi_bond_order' in result
        assert result['pi_bond_order'] > 0
        
    def test_invalid_smiles(self, predictor):
        """测试无效SMILES的错误处理"""
        with pytest.raises(MoleculeError):
            predictor.predict("invalid_smiles")
            
        with pytest.raises(MoleculeError):
            predictor.predict("")
            
        with pytest.raises(MoleculeError):
            predictor.predict("C1CC")  # 不完整的环
            
    def test_edge_cases(self, predictor):
        """测试边界情况"""
        # 单个原子
        with pytest.raises(MoleculeError):
            predictor.predict("C")
            
        # 非常长的SMILES
        long_smiles = "C" * 501
        with pytest.raises(MoleculeError):
            predictor.predict(long_smiles)
            
        # 非字符串输入
        with pytest.raises(MoleculeError):
            predictor.predict(123)
            
        
if __name__ == '__main__':
    unittest.main() 
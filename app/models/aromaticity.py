import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.Descriptors3D import (
    NPR1, NPR2, PMI1, PMI2, PMI3,
    Asphericity, Eccentricity,
    InertialShapeFactor, SpherocityIndex
)
from dataclasses import dataclass
from app.utils.errors import ModelError, MoleculeError
import logging
from app.models.geometry_optimizer import GeometryOptimizer

logger = logging.getLogger(__name__)

@dataclass
class MoleculeData:
    """分子数据的数据类"""
    hash_value: str
    smiles: str
    atom1: int
    atom2: int
    pi_bond_order: float

class MolecularFeatureExtractor:
    """分子特征提取器类"""
    
    ELECTRONEGATIVITY = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 
        'O': 3.44, 'F': 3.98, 'S': 2.58
    }
    
    def __init__(self, xyz_dir: str = "xyz-pi"):
        """初始化特征提取器
        
        Args:
            xyz_dir: XYZ文件目录路径
        """
        self.xyz_dir = Path(xyz_dir)
        self.logger = logging.getLogger('MolecularFeatureExtractor')
        
    def generate_3d_mol(self, mol: Chem.Mol, hash_value: str) -> Optional[Chem.Mol]:
        """从XYZ文件生成3D分子结构
        
        Args:
            mol: RDKit分子对象
            hash_value: 分子的哈希值
            
        Returns:
            带有3D构象的RDKit分子对象
        """
        if mol is None:
            return None
            
        mol = Chem.AddHs(mol)
        xyz_file = self.xyz_dir / f"{hash_value}.xyz"
        
        try:
            xyz_data = xyz_file.read_text().splitlines()
            xyz_num_atoms = int(xyz_data[0].strip())
            mol_num_atoms = mol.GetNumAtoms()
            
            if xyz_num_atoms != mol_num_atoms:
                self.logger.error(
                    f"原子数不匹配 - Hash: {hash_value}, "
                    f"SMILES: {mol_num_atoms}, XYZ: {xyz_num_atoms}"
                )
                return None
                
            conf = Chem.Conformer(mol_num_atoms)
            for i, line in enumerate(xyz_data[2:2+mol_num_atoms]):
                x, y, z = map(float, line.split()[1:4])
                conf.SetAtomPosition(i, (x, y, z))
                
            mol.AddConformer(conf)
            return mol
            
        except Exception as e:
            self.logger.error(f"处理XYZ文件错误 {hash_value}: {str(e)}")
            return None

    def get_atom_features(self, mol: Chem.Mol, atom_idx: int, 
                         atom_role: str) -> Tuple[List, List[str]]:
        """提取原子特征"""
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # 获取原子的环系统信息
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        rings_count = sum(1 for ring in atom_rings if atom_idx in ring)
        
        try:
            # 计算原子的Gasteiger电荷
            mol_copy = Chem.Mol(mol)
            AllChem.ComputeGasteigerCharges(mol_copy)
            gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
        except:
            gasteiger_charge = 0.0
        
        # 获取原子的邻居信息
        neighbors = atom.GetNeighbors()
        
        features = [
            # 基础原子属性
            atom.GetAtomicNum(),
            atom.GetSymbol(),
            self.ELECTRONEGATIVITY.get(atom.GetSymbol(), 0.0),
            atom.GetDegree(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),
            
            # 扩展原子属性
            atom.GetImplicitValence(),
            atom.GetExplicitValence(),
            atom.GetTotalValence(),
            atom.GetNumImplicitHs(),
            atom.GetNumExplicitHs(),
            atom.GetTotalNumHs(),
            int(atom.GetChiralTag()),
            
            # 电荷相关
            gasteiger_charge,
            atom.GetMass(),
            
            # 环系统相关
            rings_count,
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            
            # 连接性相关
            len(neighbors),
            sum(1 for n in neighbors if n.GetAtomicNum() == 6),
            sum(1 for n in neighbors if n.GetAtomicNum() == 7),
            sum(1 for n in neighbors if n.GetAtomicNum() == 8),
            sum(1 for n in neighbors if n.GetAtomicNum() == 9),
            sum(1 for n in neighbors if n.GetAtomicNum() == 16),
            sum(1 for n in neighbors if n.GetIsAromatic()),
            
            # EState特征
            Chem.rdMolDescriptors.CalcHallKierAlpha(mol),
        ]
        
        feature_names = [
            f'{atom_role}_{name}' for name in [
                'atomic_num', 'atomic_symbol', 'electronegativity',
                'degree', 'hybridization', 'is_aromatic',
                'formal_charge', 'num_radical_electrons', 'is_in_ring',
                'implicit_valence', 'explicit_valence', 'total_valence',
                'num_implicit_hs', 'num_explicit_hs', 'total_num_hs',
                'chiral_tag', 'gasteiger_charge', 'mass',
                'rings_count', 'in_ring3', 'in_ring4', 'in_ring5',
                'in_ring6', 'in_ring7', 'num_neighbors',
                'num_c_neighbors', 'num_n_neighbors', 'num_o_neighbors',
                'num_f_neighbors', 'num_s_neighbors', 'num_aromatic_neighbors',
                'hall_kier_alpha'
            ]
        ]
        
        return features, feature_names

    @staticmethod
    def get_molecular_descriptors(mol: Chem.Mol) -> Tuple[List, List[str]]:
        """提取分子描述符"""
        try:
            # 基础描述符
            basic_descriptors = [
                (Descriptors.MolWt, 'mol_weight'),
                (Descriptors.MolLogP, 'logp'),
                (Descriptors.TPSA, 'tpsa'),
                (Descriptors.ExactMolWt, 'exact_mol_weight'),
                (Descriptors.FractionCSP3, 'fraction_csp3'),
                (Descriptors.HeavyAtomMolWt, 'heavy_atom_mol_weight'),
            ]
            
            # 形状描述符
            shape_descriptors = [
                (NPR1, 'npr1'), 
                (NPR2, 'npr2'),
                (PMI1, 'pmi1'), 
                (PMI2, 'pmi2'), 
                (PMI3, 'pmi3'),
                (Asphericity, 'asphericity'),
                (Eccentricity, 'eccentricity'),
                (SpherocityIndex, 'spherocity_index'),
            ]
            
            # 环系统描述符
            ring_descriptors = [
                (Descriptors.RingCount, 'ring_count'),
                (lambda m: sum(1 for x in range(m.GetNumAtoms()) 
                             if m.GetAtomWithIdx(x).GetIsAromatic()), 'aromatic_atoms_count'),
                (Descriptors.NumAromaticRings, 'num_aromatic_rings'),
                (lambda m: sum(1 for x in Chem.GetSymmSSSR(m) 
                             if not any(m.GetAtomWithIdx(y).GetIsAromatic() for y in x)), 
                 'saturated_ring_count'),
            ]
            
            # 极性和电荷相关描述符
            charge_descriptors = [
                (Descriptors.NumValenceElectrons, 'num_valence_electrons'),
                (lambda m: max(float(x.GetProp('_GasteigerCharge')) 
                             for x in m.GetAtoms() if '_GasteigerCharge' in x.GetPropsAsDict()), 
                 'max_partial_charge'),
                (lambda m: min(float(x.GetProp('_GasteigerCharge')) 
                             for x in m.GetAtoms() if '_GasteigerCharge' in x.GetPropsAsDict()), 
                 'min_partial_charge'),
            ]
            
            # 连接性描述符
            connectivity_descriptors = [
                (Descriptors.Chi0n, 'chi0n'),
                (Descriptors.Chi1n, 'chi1n'),
                (Descriptors.Chi2n, 'chi2n'),
                (Descriptors.Chi3n, 'chi3n'),
                (Descriptors.Chi4n, 'chi4n'),
            ]
            
            # 复杂性描述符
            complexity_descriptors = [
                (Descriptors.BertzCT, 'bertz_complexity'),
            ]
            
            # 合并所有描述符
            descriptors = (
                basic_descriptors + shape_descriptors + ring_descriptors +
                charge_descriptors + connectivity_descriptors +
                complexity_descriptors
            )
            
            # 预先计算Gasteiger电荷
            AllChem.ComputeGasteigerCharges(mol)
            
            # 计算所有描述符
            features = []
            feature_names = []
            
            for desc_func, name in descriptors:
                try:
                    value = desc_func(mol) if callable(desc_func) else 0
                    features.append(float(value))
                    feature_names.append(name)
                except Exception as e:
                    features.append(0.0)
                    feature_names.append(name)
            
            return features, feature_names
            
        except Exception as e:
            logger.error(f"计算分子描述符时出错: {str(e)}")
            features = [0.0] * len(descriptors)
            feature_names = [name for _, name in descriptors]
            return features, feature_names

# 保留原有的全局预测器实例和相关函数
_predictor = None

def init_model(xyz_dir: str = "xyz-pi") -> None:
    """初始化全局预测器实例"""
    global _predictor
    _predictor = AromaticityPredictor(xyz_dir)

def predict_aromaticity(smiles: str, hash_value: str) -> Dict[str, Any]:
    """预测分子的π键级
    
    Args:
        smiles: 分子的SMILES字符串
        hash_value: 分子的哈希值
    """
    global _predictor
    if _predictor is None:
        init_model()
    return _predictor.predict(smiles, hash_value)

class AromaticityPredictor:
    def __init__(self, xyz_dir: str = "xyz-pi"):
        """初始化预测器"""
        try:
            self.model_path = Path(__file__).parent / "models" / "rf_aromaticity_predictor.joblib"
            self.scaler_path = Path(__file__).parent / "models" / "feature_scaler.joblib"
            
            self._model = joblib.load(self.model_path)
            self._scaler = joblib.load(self.scaler_path)
            self.feature_extractor = MolecularFeatureExtractor(xyz_dir)
            self.optimizer = GeometryOptimizer()
            self.xyz_dir = Path(xyz_dir)
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise ModelError(f"模型初始化失败: {str(e)}")

    def predict(self, smiles: str, hash_value: str) -> Dict[str, Any]:
        """预测分子的π键级
        
        Args:
            smiles: 分子的SMILES字符串
            hash_value: 分子的哈希值，用于查找对应的XYZ文件
            
        Returns:
            包含预测结果的字典
        """
        try:
            # 输入验证
            if not isinstance(smiles, str):
                raise MoleculeError("SMILES必须是字符串")
            if not smiles.strip():
                raise MoleculeError("SMILES不能为空")
            if len(smiles) > 500:
                raise MoleculeError("SMILES太长")
                
            # 转换SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise MoleculeError(f"无效的SMILES: {smiles}")
            if mol.GetNumAtoms() == 0:
                raise MoleculeError("分子不能没有原子")
                
            # 从XYZ文件生成3D构象
            mol = self.optimize_if_needed(mol, hash_value)
            if mol is None:
                raise MoleculeError(f"无法从XYZ文件生成3D构象: {hash_value}")
            
            # 提取特征
            mol_features, _ = self.feature_extractor.get_molecular_descriptors(mol)
            maccs = list(GetMACCSKeysFingerprint(mol))
            
            # 合并特征
            features = np.concatenate([mol_features, maccs])
            features = features.reshape(1, -1)
            
            # 标准化
            features = self._scaler.transform(features)
            
            # 预测π键级
            pi_bond_order = float(self._model.predict(features)[0])
            
            return {
                'pi_bond_order': pi_bond_order,
                'aromatic_rings': self._find_aromatic_rings(mol)
            }
            
        except MoleculeError as e:
            raise e
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise ModelError(f"预测失败: {str(e)}")
            
    def _find_aromatic_rings(self, mol: Chem.Mol) -> List[List[int]]:
        """查找芳香环"""
        try:
            rings = []
            for ring in mol.GetRingInfo().AtomRings():
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    rings.append(list(ring))
            return rings
        except Exception as e:
            logger.error(f"查找芳香环失败: {str(e)}")
            return []

    def optimize_if_needed(self, mol: Chem.Mol, hash_value: str) -> Optional[Chem.Mol]:
        """如果XYZ文件不存在，则进行优化"""
        xyz_file = self.xyz_dir / f"{hash_value}.xyz"
        if not xyz_file.exists():
            logger.info(f"XYZ文件不存在，进行几何优化: {hash_value}")
            if self.optimizer.optimize_and_save(mol, hash_value, self.xyz_dir):
                return self.generate_3d_mol(mol, hash_value)
        return self.generate_3d_mol(mol, hash_value)

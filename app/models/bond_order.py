import json
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.Descriptors3D import (NPR1, NPR2, PMI1, PMI2, PMI3, Asphericity, 
                                      Eccentricity, InertialShapeFactor, SpherocityIndex)
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import logging
from functools import lru_cache
import hashlib

from app.utils.errors import ModelError, MoleculeError, CarbeneError
from app.models.geometry_optimizer import GeometryOptimizer

logger = logging.getLogger(__name__)

pauling_electronegativity = {
    1: 2.20,  # H
    6: 2.55,  # C
    7: 3.04,  # N
    8: 3.44,  # O
    9: 3.98,  # F
    16: 2.58, # S
    17: 3.16, # Cl
    # 必要时加入更多元素
}


def read_xyz(file_path):
    """读取XYZ文件，确保读取所有原子包括氢原子"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # 第一行是原子数
        num_atoms = int(lines[0].strip())
        
        # 验证文件格式
        if len(lines) < num_atoms + 2:
            raise ValueError(f"XYZ文件格式错误：期望{num_atoms + 2}行，实际{len(lines)}行")
            
        atoms = []
        coords = []
        
        # 从第三行开始读取原子坐标（跳过原子数和注释行）
        for line in lines[2:2+num_atoms]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
            
        if len(atoms) != num_atoms:
            raise ValueError(f"原子数不匹配：期望{num_atoms}个原子，实际读取{len(atoms)}个原子")
            
        return atoms, np.array(coords)
    except Exception as e:
        print(f"读取XYZ文件失败: {str(e)}")
        return None, None

def calculate_distance(coords, idx1, idx2):
    return np.linalg.norm(coords[idx1] - coords[idx2])

def calculate_angle(coords, idx1, idx2, idx3):
    v1 = coords[idx1] - coords[idx2]
    v2 = coords[idx3] - coords[idx2]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_mapping_score(mol, current_indices, original_indices):
    """计算卡宾碳的评分
    
    条件：
    1. 必须是芳香碳
    2. 只能有两个重原子邻居（非氢原子）
    3. 必须在五元环中
    """
    c_idx, n1_idx, n2_idx = current_indices
    carbene = mol.GetAtomWithIdx(c_idx)
    n1 = mol.GetAtomWithIdx(n1_idx)
    n2 = mol.GetAtomWithIdx(n2_idx)
    
    # 检查是否是芳香碳
    if carbene.GetSymbol() != 'C' or not carbene.GetIsAromatic():
        return -1
        
    # 检查重原子邻居数量
    heavy_neighbors = [n for n in carbene.GetNeighbors() if n.GetSymbol() != 'H']
    if len(heavy_neighbors) != 2:
        return -1
        
    # 检查是否在五元环中
    if not carbene.IsInRingSize(5):
        return -1
        
    # 检查邻居原子类型
    neighbor_symbols = {n1.GetSymbol(), n2.GetSymbol()}
    if neighbor_symbols not in [{'N', 'S'}, {'N', 'N'}]:
        return -1
        
    return 1

def validate_and_remap_carbene_structure(mol, molecule_data, index_offset=1):
    """验证并重映射卡宾结构
    
    验证条件：
    1. 必须是芳香碳
    2. 只能有两个重原子邻居
    3. 必须在五元环中
    4. 邻居必须是特定组合（N,N或N,S）
    """
    hash_value = molecule_data['Hash']
    original_carbene_idx = molecule_data['First_Atom_Index'] - index_offset
    original_n1_idx = molecule_data['Neighbor_1_Index'] - index_offset 
    original_n2_idx = molecule_data['Neighbor_2_Index'] - index_offset

    # 只考虑芳香碳原子
    carbon_indices = [atom.GetIdx() for atom in mol.GetAtoms() 
                     if atom.GetSymbol() == 'C' and atom.GetIsAromatic()]
    
    best_match = None
    best_score = -1
    mapped_indices = {}

    for c_idx in carbon_indices:
        carbene_atom = mol.GetAtomWithIdx(c_idx)
        # 获取重原子邻居
        heavy_neighbors = [n.GetIdx() for n in carbene_atom.GetNeighbors() 
                         if n.GetSymbol() != 'H']
        
        # 检查重原子邻居数量
        if len(heavy_neighbors) != 2:
            continue
            
        n1_idx, n2_idx = heavy_neighbors
        score = calculate_mapping_score(mol, 
                                     (c_idx, n1_idx, n2_idx),
                                     (original_carbene_idx, original_n1_idx, original_n2_idx))
        
        if score > best_score:
            best_score = score
            best_match = (c_idx, n1_idx, n2_idx)
            mapped_indices = {
                'carbene_idx': c_idx,
                'neighbor1_idx': n1_idx,
                'neighbor2_idx': n2_idx
            }

    if best_match:
        print(f"\nRemapped indices for {hash_value}:")
        print(f"Original: carbene={original_carbene_idx+1}, n1={original_n1_idx+1}, n2={original_n2_idx+1}")
        print(f"Remapped: carbene={mapped_indices['carbene_idx']+1}, "
              f"n1={mapped_indices['neighbor1_idx']+1}, "
              f"n2={mapped_indices['neighbor2_idx']+1}")
              
        # 打印详细信息
        carbene = mol.GetAtomWithIdx(mapped_indices['carbene_idx'])
        n1 = mol.GetAtomWithIdx(mapped_indices['neighbor1_idx'])
        n2 = mol.GetAtomWithIdx(mapped_indices['neighbor2_idx'])
        print(f"\n卡宾碳详细信息:")
        print(f"位置: {mapped_indices['carbene_idx']+1}")
        print(f"元素: {carbene.GetSymbol()}")
        print(f"芳香性: {carbene.GetIsAromatic()}")
        print(f"邻居1: {n1.GetSymbol()} (位置 {mapped_indices['neighbor1_idx']+1})")
        print(f"邻居2: {n2.GetSymbol()} (位置 {mapped_indices['neighbor2_idx']+1})")
        print(f"在五元环中: {carbene.IsInRingSize(5)}")
        
        return True, mapped_indices
    
    return False, None

def get_bond_features(bond, mol):
    if bond is None:
        return [0, 0, 0, 0, 0]
    bond_type = bond.GetBondTypeAsDouble()
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    stereo = int(bond.GetStereo())
    electronegativity_diff = abs(
        pauling_electronegativity[mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum()] -
        pauling_electronegativity[mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum()]
    )
    return [bond_type, is_conjugated, is_in_ring, stereo, electronegativity_diff]

def get_atom_features(mol, atom_idx, coords):
    if atom_idx >= mol.GetNumAtoms():
        return [0]*22
    atom = mol.GetAtomWithIdx(atom_idx)
    
    hybridization_map = {
        Chem.rdchem.HybridizationType.S: 1,    
        Chem.rdchem.HybridizationType.SP: 2,   
        Chem.rdchem.HybridizationType.SP2: 3,  
        Chem.rdchem.HybridizationType.SP3: 4,  
        Chem.rdchem.HybridizationType.SP3D: 5, 
        Chem.rdchem.HybridizationType.SP3D2: 6,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0  
    }
    chiral_map = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.rdchem.ChiralType.CHI_OTHER: 3
    }
    features = [
        atom.GetAtomicNum(),
        pauling_electronegativity.get(atom.GetAtomicNum(), 0.0),
        atom.GetDegree(),
        hybridization_map[atom.GetHybridization()],
        int(atom.GetIsAromatic()),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        int(atom.IsInRing()),
        atom.GetImplicitValence(),
        atom.GetExplicitValence(),
        atom.GetTotalValence(),
        atom.GetTotalNumHs(),
        chiral_map[atom.GetChiralTag()],
        atom.GetMass(),
        len(atom.GetNeighbors()),
        len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'C']),
        len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'N']),
        len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'O']),
        len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'F']),
        len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'S']),
        len([n for n in atom.GetNeighbors() if n.GetIsAromatic()]),
        sum(1 for r in mol.GetRingInfo().AtomRings() if atom_idx in r)
    ]
    return features

def extract_features_from_mol(mol, coords, carbene_idx, neighbor1_idx, neighbor2_idx):
    features = []
    carbene_features = get_atom_features(mol, carbene_idx, coords)
    neighbor1_features = get_atom_features(mol, neighbor1_idx, coords)
    neighbor2_features = get_atom_features(mol, neighbor2_idx, coords)
    features.extend(carbene_features)
    features.extend(neighbor1_features)
    features.extend(neighbor2_features)
    
    bond1 = mol.GetBondBetweenAtoms(carbene_idx, neighbor1_idx)
    bond2 = mol.GetBondBetweenAtoms(carbene_idx, neighbor2_idx)
    features.extend(get_bond_features(bond1, mol))
    features.extend(get_bond_features(bond2, mol))
    
    dist1 = calculate_distance(coords, carbene_idx, neighbor1_idx)
    dist2 = calculate_distance(coords, carbene_idx, neighbor2_idx)
    angle = calculate_angle(coords, neighbor1_idx, carbene_idx, neighbor2_idx)
    features.extend([dist1, dist2, angle])
    
    shape_descriptors = [
        NPR1(mol), NPR2(mol), PMI1(mol), PMI2(mol), PMI3(mol),
        Asphericity(mol), Eccentricity(mol), InertialShapeFactor(mol),
        SpherocityIndex(mol)
    ]
    features.extend(shape_descriptors)
    
    mol_descriptors = [
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        Descriptors.ExactMolWt(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) else 0.0,
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol)
    ]
    features.extend(mol_descriptors)
    
    maccs_fp = GetMACCSKeysFingerprint(mol)
    features.extend(list(map(int, maccs_fp.ToBitString())))
    
    return np.array(features)

def process_molecule(molecule_data, xyz_folder, index_offset=1):
    hash_value = molecule_data['Hash']
    xyz_file_path = os.path.join(xyz_folder, f"{hash_value}.xyz")
    if not os.path.exists(xyz_file_path):
        print(f"Warning: File {xyz_file_path} not found.")
        return None

    smiles = molecule_data['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Could not create molecule from SMILES: {smiles}")
        return None

    try:
        # 使用非kekulize方式sanitize
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE)
        Chem.AssignStereochemistry(mol)
        # 尝试kekulize, 失败则跳过
        try:
            Chem.Kekulize(mol)
        except Chem.rdchem.KekulizeException:
            pass
    except Exception as e:
        print(f"Error: Molecule processing failed: {str(e)}")
        return None

    try:
        mol = Chem.AddHs(mol)
        print(f"\n添加氢原子后的原子数: {mol.GetNumAtoms()}")
    except Exception as e:
        print(f"Error: Could not add hydrogens: {str(e)}")
        return None

    try:
        carbene_idx = molecule_data['First_Atom_Index'] - index_offset
        carbene_atom = mol.GetAtomWithIdx(carbene_idx)
        carbene_atom.SetNumRadicalElectrons(2)
        carbene_atom.SetNoImplicit(True)
    except Exception as e:
        print(f"Error: Could not process carbene carbon: {str(e)}")
        return None

    try:
        atoms, coords = read_xyz(xyz_file_path)
        if atoms is None or coords is None:
            print("Error: Could not read XYZ file properly")
            return None
            
        print(f"\nAnalyzing molecule {hash_value}:")
        print(f"SMILES: {molecule_data['SMILES']}")
        print(f"XYZ file atom count: {len(atoms)}")
        print(f"Molecule atom count: {mol.GetNumAtoms()}")
        
        if mol.GetNumAtoms() != len(atoms):
            print(f"Error: Atom count mismatch - SMILES: {mol.GetNumAtoms()}, XYZ: {len(atoms)}")
            return None
    except Exception as e:
        print(f"Error: Could not read XYZ file: {str(e)}")
        return None

    try:
        valid, mapped_indices = validate_and_remap_carbene_structure(mol, molecule_data, index_offset)
        if not valid:
            print("Error: Could not validate or remap carbene structure")
            return None
        carbene_idx = mapped_indices['carbene_idx']
        neighbor1_idx = mapped_indices['neighbor1_idx']
        neighbor2_idx = mapped_indices['neighbor2_idx']
    except Exception as e:
        print(f"Error: Could not validate carbene structure: {str(e)}")
        return None

    try:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(coords):
            conf.SetAtomPosition(i, coord)
        mol.AddConformer(conf)
    except Exception as e:
        print(f"Error adding coordinates: {str(e)}")
        return None

    try:
        features = extract_features_from_mol(mol, coords, carbene_idx, neighbor1_idx, neighbor2_idx)
        if 'bond_order1' in molecule_data and 'bond_order2' in molecule_data:
            features = np.append(features, [
                molecule_data['bond_order1'],
                molecule_data['bond_order2']
            ])
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

    print("Successfully processed molecule")
    return features

class BondOrderModel:
    def __init__(self, xyz_dir: str = "xyz-carbene"):
        try:
            self.xyz_dir = Path(xyz_dir)
            model_path = Path(__file__).parent / "models" / "rf_bond_order_predictor.joblib"
            model_dict = joblib.load(model_path)
            self.model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
            self.scaler = None
            self.optimizer = GeometryOptimizer()
            logger.info("功加载键级预测模型")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise ModelError(f"模型加载失败: {str(e)}")
            
    def predict(self, smiles: str, carbene_idx: int, hash_value: str) -> Dict[str, Any]:
        try:
            if not isinstance(smiles, str):
                raise MoleculeError("SMILES必须是字符串")
            if not smiles.strip():
                raise MoleculeError("SMILES不能为空")
            if len(smiles) > 500:
                raise MoleculeError("SMILES太长")

            # 创建分子对象来获取邻居信息
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise MoleculeError("无法从SMILES创建分子")
            
            # 获取卡宾原子的邻居
            carbene_atom = mol.GetAtomWithIdx(carbene_idx)
            neighbors = [n.GetIdx() for n in carbene_atom.GetNeighbors()]
            if len(neighbors) < 2:
                raise CarbeneError("卡宾原子必须有至少两个邻居")

            molecule_data = {
                'SMILES': smiles,
                'Hash': hash_value,
                'First_Atom_Index': carbene_idx + 1,
                'Neighbor_1_Index': neighbors[0] + 1,
                'Neighbor_2_Index': neighbors[1] + 1
            }

            features = process_molecule(molecule_data, self.xyz_dir)
            if features is None:
                raise MoleculeError(f"特征提取失败: {hash_value}")

            features = features.reshape(1, -1)
            predictions = self.model.predict(features)
            
            return {
                'bond_orders': {0: float(predictions[0])},
                'confidence': 0.9
            }
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise ModelError(f"预测失败: {str(e)}")

class BondOrderPredictor:
    def __init__(self, xyz_dir: Optional[str] = "xyz-carbene"):
        try:
            self.model = BondOrderModel(xyz_dir)
            self.feature_names = None
            self.scaler = None
            self.cache = {}
        except Exception as e:
            logger.error(f"预测器初始化失败: {str(e)}")
            raise ModelError(f"预测器初始化失败: {str(e)}")

    def predict(self, smiles: str, carbene_idx: int, hash_value: str) -> Dict[str, Any]:
        return self.model.predict(smiles, carbene_idx, hash_value)

predictor = None

def init_model(xyz_dir: Optional[str] = "xyz-carbene") -> None:
    global predictor
    try:
        predictor = BondOrderPredictor(xyz_dir)
    except Exception as e:
        logger.error(f"初始化预测器失败: {str(e)}")
        raise ModelError(f"初始化预测器失败: {str(e)}")

def generate_xyz_file(smiles: str, hash_value: str, xyz_dir: str = "xyz-carbene") -> str:
    """从SMILES生成XYZ文件
    
    Args:
        smiles: SMILES字符串
        hash_value: 分子哈希值
        xyz_dir: XYZ文件目录
        
    Returns:
        XYZ文件路径
    """
    try:
        # 生成3D构象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("无效的SMILES字符串")
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # 创建XYZ文件
        xyz_path = os.path.join(xyz_dir, f"{hash_value}.xyz")
        
        # 获取原子坐标
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        
        # 写入XYZ文件
        with open(xyz_path, 'w') as f:
            f.write(f"{mol.GetNumAtoms()}\n")  # 原子数
            f.write(f"Generated from SMILES: {smiles}\n")  # 注释行
            
            for i, atom in enumerate(mol.GetAtoms()):
                pos = coords[i]
                f.write(f"{atom.GetSymbol()} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                
        return xyz_path
    except Exception as e:
        raise MoleculeError(f"生成XYZ文件失败: {str(e)}")

@lru_cache(maxsize=1000)
def predict_bond_order(smiles: str, carbene_idx: int, hash_value: str) -> Dict[str, Any]:
    """预测分子的键级
    
    Args:
        smiles: SMILES字符串
        carbene_idx: 卡宾原子索引
        hash_value: 分子哈希值
        
    Returns:
        预测结果字典
    """
    if predictor is None:
        raise ModelError("预测器未初始化")
        
    try:
        # 生成XYZ文件
        xyz_path = generate_xyz_file(smiles, hash_value)
        
        # 调用预测器
        result = predictor.predict(smiles, carbene_idx, hash_value)
        
        # 清理XYZ文件
        try:
            os.remove(xyz_path)
        except:
            pass
            
        return result
    except Exception as e:
        raise ModelError(f"预测失败: {str(e)}")

def predict_bond_orders_batch(molecules: List[Dict[str, Any]], n_jobs: int = -1) -> List[Dict[str, Any]]:
    """批量预测分子的键级
    
    Args:
        molecules: 分子列表，每个分子是包含smiles和carbene_idx的字典
        n_jobs: 并行任务数，-1表示使用所有CPU核心
        
    Returns:
        预测结果列表
    """
    results = []
    for mol in molecules:
        try:
            # 生成哈希值
            hash_value = hashlib.md5(mol['smiles'].encode()).hexdigest()
            
            # 预测单个分子
            result = predict_bond_order(
                smiles=mol['smiles'],
                carbene_idx=mol['carbene_idx'],
                hash_value=hash_value
            )
            results.append({
                'smiles': mol['smiles'],
                'carbene_idx': mol['carbene_idx'],
                'bond_orders': result['bond_orders']
            })
        except Exception as e:
            results.append({
                'smiles': mol['smiles'],
                'carbene_idx': mol['carbene_idx'],
                'error': str(e)
            })
            
    return results

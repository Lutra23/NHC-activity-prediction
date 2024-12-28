from rdkit import Chem

def validate_smiles(smiles):
    """验证SMILES字符串是否有效"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False 
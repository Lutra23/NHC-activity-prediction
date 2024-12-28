from marshmallow import Schema, fields, validate, ValidationError as MarshmallowError
from functools import wraps
from flask import request, jsonify
from .errors import ValidationError
from rdkit import Chem

def validate_smiles(smiles: str) -> bool:
    """验证SMILES字符串
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        是否有效
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def validate_atom_index(smiles: str, atom_idx: int) -> bool:
    """验证原子索引是否有效
    
    Args:
        smiles: SMILES字符串
        atom_idx: 原子索引
        
    Returns:
        是否有效
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return 0 <= atom_idx < mol.GetNumAtoms()
    except:
        return False

class MoleculeSchema(Schema):
    """分子数据的验证模式"""
    smiles = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=1000),
        error_messages={"required": "SMILES字符串是必需的"}
    )
    carbene_idx = fields.Int(
        required=True,
        validate=validate.Range(min=0),
        error_messages={"required": "卡宾碳原子索引是必需的"}
    )

class PredictionSchema(Schema):
    """预测请求的验证模式"""
    smiles = fields.Str(
        required=True,
        validate=[
            validate.Length(min=1, max=1000),
            lambda s: validate_smiles(s) or "无效的SMILES字符串"
        ],
        error_messages={"required": "SMILES字符串是必需的"}
    )
    carbene_idx = fields.Int(
        required=True,
        validate=[
            validate.Range(min=0),
            lambda idx, ctx: validate_atom_index(ctx['smiles'], idx) or "无效的原子索引"
        ],
        error_messages={"required": "卡宾碳原子索引是必需的"}
    )
    options = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False
    )

class BatchPredictionSchema(Schema):
    """批量预测请求的验证模式"""
    molecules = fields.List(
        fields.Nested(MoleculeSchema),
        required=True,
        validate=validate.Length(min=1, max=100),
        error_messages={
            "required": "分子列表是必需的",
            "length": "分子列表长度必须在1到100之间"
        }
    )
    n_jobs = fields.Int(
        required=False,
        validate=validate.Range(min=-1),
        load_default=-1
    )
    options = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False
    )

def validate_request(schema_class):
    """请求验证装饰器
    
    Args:
        schema_class: Schema类
        
    Returns:
        装饰器函数
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            schema = schema_class()
            try:
                # 获取请求数据
                if request.is_json:
                    data = request.get_json()
                elif request.form:
                    data = request.form
                else:
                    data = request.args

                # 验证数据
                validated_data = schema.load(data)
                
                # 将验证后的数据传递给视图函数
                return f(validated_data, *args, **kwargs)
                
            except MarshmallowError as e:
                # 转换验证错误为API错误
                raise ValidationError(
                    message="输入验证失败",
                    details=e.messages
                )
                
        return decorated_function
    return decorator 
from flask import Blueprint, jsonify, request
from app.models.bond_order import predict_bond_order, predict_batch
from app.models.aromaticity import predict_aromaticity
from app.utils.validation import validate_request, PredictionSchema, BatchPredictionSchema
from app.utils.errors import ModelError, ValidationError
from app.utils.middleware import rate_limit
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
bp = Blueprint('api', __name__)

@bp.route('/predict/carbene-bond', methods=['POST'])
@rate_limit
@validate_request(PredictionSchema)
def predict_carbene_bond(data: Dict[str, Any]):
    """预测卡宾键级
    
    Args:
        data: 经过验证的请求数据,包含:
            - smiles: 分子的SMILES表示
            - carbene_idx: 卡宾碳原子索引
            - neighbor1_idx: (可选)第一个邻近原子索引
            - neighbor2_idx: (可选)第二个邻近原子索引
        
    Returns:
        JSON响应，包含预测结果
    """
    try:
        # 验证必需参数
        if 'carbene_idx' not in data:
            raise ValidationError("缺少必需参数: carbene_idx")
            
        carbene_idx = int(data['carbene_idx'])
        
        # 预测
        result = predict_bond_order(
            smiles=data['smiles'],
            carbene_idx=carbene_idx
        )
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except ValidationError as e:
        raise e
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"预测卡宾键级失败: {str(e)}")
        raise ModelError(
            message="预测卡宾键级失败",
            details={'error': str(e)}
        )

@bp.route('/predict/carbene-bond/batch', methods=['POST'])
@rate_limit
@validate_request(BatchPredictionSchema)
def predict_carbene_bond_batch(data: Dict[str, Any]):
    """批量预测卡宾键级
    
    Args:
        data: 经过验证的请求数据,包含:
            - molecules: 分子列表,每个分子包含:
                - smiles: 分子的SMILES表示
                - carbene_idx: 卡宾碳原子索引
            - n_jobs: (可选)并行作业数,-1表示使用所有CPU
        
    Returns:
        JSON响应，包含所有预测结果
    """
    try:
        # 验证必需参数
        if 'molecules' not in data:
            raise ValidationError("缺少必需参数: molecules")
            
        molecules = data['molecules']
        if not isinstance(molecules, list):
            raise ValidationError("molecules必须是列表")
            
        # 提取预测参数
        smiles_list = []
        for mol in molecules:
            if 'smiles' not in mol or 'carbene_idx' not in mol:
                raise ValidationError("每个分子必须包含smiles和carbene_idx")
            smiles_list.append((mol['smiles'], int(mol['carbene_idx'])))
        
        # 获取并行作业数
        n_jobs = data.get('n_jobs', -1)
        
        # 批量预测
        results = predict_batch(smiles_list, n_jobs=n_jobs)
        
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': results,
                'total': len(results),
                'successful': len([r for r in results if r is not None])
            }
        })
        
    except ValidationError as e:
        raise e
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"批量预测卡宾键级失败: {str(e)}")
        raise ModelError(
            message="批量预测卡宾键级失败",
            details={'error': str(e)}
        )

@bp.route('/predict/aromaticity', methods=['POST'])
@rate_limit
@validate_request(PredictionSchema)
def predict_aromatic(data: Dict[str, Any]):
    """预测芳香性
    
    Args:
        data: 经过验证的请求数据
        
    Returns:
        JSON响应，包含预测结果
    """
    try:
        result = predict_aromaticity(data['smiles'])
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        logger.error(f"预测芳香性失败: {str(e)}")
        raise ModelError(
            message="预测芳香性失败",
            details={'error': str(e)}
        )

@bp.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    from app.models.bond_order import predictor
    
    return jsonify({
        'status': 'success',
        'message': 'Service is healthy',
        'model_info': {
            'bond_order_model': {
                'loaded': predictor.model is not None,
                'feature_count': len(predictor.feature_names) if predictor.feature_names else 0
            }
        },
        'cache_info': {
            'bond_order_cache_size': len(predictor.cache),
            'aromaticity_cache_size': len(predict_aromaticity.cache)
        }
    }) 
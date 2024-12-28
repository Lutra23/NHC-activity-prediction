from flask import Blueprint, render_template, jsonify, request
from rdkit import Chem
import hashlib

# 创建蓝图
bp = Blueprint('main', __name__)

# Web界面路由
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/batch')
def batch():
    return render_template('batch.html')

# API路由
@bp.route('/api/v1/health')
def health_check():
    return jsonify({'status': 'ok'})

@bp.route('/api/v1/predict/carbene-bond', methods=['POST'])
def predict_carbene_bond():
    data = request.get_json()
    
    if not data or 'smiles' not in data or 'carbene_idx' not in data:
        return jsonify({
            'error': '请提供SMILES字符串和卡宾原子索引'
        }), 400
    
    try:
        # 生成分子哈希值
        mol = Chem.MolFromSmiles(data['smiles'])
        if mol is None:
            return jsonify({
                'error': 'SMILES字符串无效'
            }), 400
        
        # 使用SMILES字符串生成哈希值
        hash_value = hashlib.md5(data['smiles'].encode()).hexdigest()
        
        # 调用预测模型
        from app.models.bond_order import predict_bond_order, init_model
        
        # 确保模型已初始化
        try:
            init_model()
        except Exception as e:
            pass  # 如果已经初始化，会抛出异常，我们可以忽略
            
        result = predict_bond_order(
            smiles=data['smiles'],
            carbene_idx=data['carbene_idx'],
            hash_value=hash_value
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@bp.route('/api/v1/predict/carbene-bond/batch', methods=['POST'])
def predict_carbene_bond_batch():
    data = request.get_json()
    
    if not data or 'molecules' not in data:
        return jsonify({
            'error': '请提供分子列表'
        }), 400
    
    try:
        # 导入必要的模块
        from app.models.bond_order import predict_bond_orders_batch, init_model
        
        # 确保模型已初始化
        try:
            init_model()
        except Exception as e:
            pass  # 如果已经初始化，会抛出异常，我们可以忽略
            
        # 这里调用批量预测
        results = predict_bond_orders_batch(
            data['molecules'],
            n_jobs=data.get('n_jobs', -1)
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500 
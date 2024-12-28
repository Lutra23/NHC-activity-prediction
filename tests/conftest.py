import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置测试环境变量
os.environ['FLASK_ENV'] = 'testing'
os.environ['TESTING'] = 'true'

import pytest
from app import create_app
from app.config import TestingConfig

@pytest.fixture
def app():
    """创建测试应用实例"""
    app = create_app(TestingConfig)
    return app

@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """创建CLI测试运行器"""
    return app.test_cli_runner()

# 测试数据
@pytest.fixture
def test_smiles():
    """测试用SMILES"""
    return "CC(=O)O"  # 乙酸

@pytest.fixture
def test_carbene_smiles():
    """测试用卡宾SMILES"""
    return "C[C]"  # 甲基卡宾

@pytest.fixture
def invalid_smiles():
    """无效的SMILES"""
    return "invalid_smiles"

@pytest.fixture
def test_batch_data():
    """批量测试数据"""
    return {
        "molecules": [
            {"smiles": "CC(=O)O", "carbene_idx": 1},
            {"smiles": "CCO", "carbene_idx": 0}
        ],
        "n_jobs": -1
    } 
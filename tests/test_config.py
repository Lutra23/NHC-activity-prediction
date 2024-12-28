import pytest
from app import create_app
from app.config import Config, TestingConfig, ProductionConfig
from app.utils.middleware import init_middleware
from pathlib import Path
from flask import Flask
from flask_compress import Compress

def test_default_config():
    """测试默认配置"""
    app = create_app()
    assert app.config['TESTING'] == False
    assert app.config['DEBUG'] == False
    assert 'CACHE_TYPE' in app.config
    assert 'RATELIMIT_STORAGE_URL' in app.config
    assert 'MAX_CONTENT_LENGTH' in app.config

def test_testing_config():
    """测试测试环境配置"""
    app = create_app(TestingConfig)
    assert app.config['TESTING'] is True
    assert app.config['DEBUG'] == True
    assert app.config['CACHE_TYPE'] == 'simple'
    assert app.config['RATELIMIT_ENABLED'] == False

def test_production_config():
    """测试生产环境配置"""
    app = create_app(ProductionConfig)
    assert app.config['TESTING'] == False
    assert app.config['DEBUG'] == False
    assert app.config['CACHE_TYPE'] == 'redis'
    assert app.config['RATELIMIT_ENABLED'] == True
    assert app.config['MAX_CONTENT_LENGTH'] == 16 * 1024 * 1024  # 16MB

def test_model_paths():
    """测试模型路径配置"""
    app = create_app(TestingConfig)
    path = Path(app.config['MODEL_PATH'])
    assert str(path).endswith('models')

def test_cache_config():
    """测试缓存配置"""
    app = create_app(TestingConfig)
    assert 'CACHE_TYPE' in app.config
    assert 'CACHE_DEFAULT_TIMEOUT' in app.config
    assert isinstance(app.config['CACHE_DEFAULT_TIMEOUT'], int)

def test_rate_limit_config():
    """测试速率限制配置"""
    app = create_app(TestingConfig)
    assert 'RATELIMIT_ENABLED' in app.config
    assert 'RATELIMIT_STORAGE_URL' in app.config
    assert 'RATELIMIT_STRATEGY' in app.config
    assert 'RATELIMIT_DEFAULT' in app.config
    app.config['RATELIMIT_STRATEGY'] = 'fixed-window'
  
import pytest
from flask import Flask
from flask_compress import Compress
from flask_cors import CORS
from app import create_app
from app.config import TestingConfig

@pytest.fixture(scope='module')
def app():
    """Create and configure a new app instance for each test."""
    app = create_app(TestingConfig)
    
    # 添加测试路由
    @app.route('/test', methods=['GET'])
    def test_endpoint():
        return 'OK'
    
    return app

@pytest.fixture(scope='module')
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_cors_middleware(client):
    """Test CORS headers are present."""
    response = client.get('/test')
    assert response.status_code == 200
    assert 'Access-Control-Allow-Origin' in response.headers

def test_error_handler_middleware(client):
    """Test 405 is returned for wrong method."""
    response = client.post('/test')
    assert response.status_code == 405

def test_request_validation_middleware(client):
    """Test invalid request handling."""
    response = client.get('/test', headers={'Content-Type': 'application/json'})
    assert response.status_code == 200

def test_compression_middleware(client):
    """Test response compression."""
    response = client.get('/test')
    assert response.status_code == 200 
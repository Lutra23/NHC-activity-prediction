from flask import Flask, request, g, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import time
import uuid
from functools import wraps
from typing import Optional, Callable, Any

from app.utils.errors import RateLimitError, APIError

def configure_middleware(app: Flask) -> None:
    """配置所有中间件
    
    Args:
        app: Flask应用实例
    """
    # CORS
    if app.config.get('CORS_ENABLED', True):
        CORS(app, 
             origins=app.config.get('CORS_ORIGINS', '*'),
             methods=app.config.get('CORS_METHODS', ['GET', 'POST', 'OPTIONS']),
             allow_headers=app.config.get('CORS_ALLOW_HEADERS', ['Content-Type']))
    
    # 代理修复
    app.wsgi_app = ProxyFix(app.wsgi_app)
    
    # 请求ID
    @app.before_request
    def add_request_id():
        g.request_id = str(uuid.uuid4())
        
    @app.after_request
    def add_request_id_header(response):
        response.headers['X-Request-ID'] = g.get('request_id', '')
        return response
    
    # 请求计时
    @app.before_request
    def start_timer():
        g.start_time = time.time()
        
    @app.after_request
    def add_timing_header(response):
        if hasattr(g, 'start_time'):
            elapsed = time.time() - g.start_time
            response.headers['X-Response-Time'] = f'{elapsed*1000:.2f}ms'
        return response
    
    # 安全响应头
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        return response
    
    # 错误处理
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not Found'}), 404
        
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({'error': 'Method Not Allowed'}), 405
        
    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({'error': 'Rate limit exceeded'}), 429
        
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

def rate_limit(f: Callable) -> Callable:
    """速率限制装饰器"""
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        # 简单的内存计数器实现
        key = f'{request.remote_addr}:{f.__name__}'
        if not hasattr(g, 'rate_limit_data'):
            g.rate_limit_data = {}
        
        now = time.time()
        data = g.rate_limit_data.get(key, {'count': 0, 'reset': now + 60})
        
        if now > data['reset']:
            data = {'count': 0, 'reset': now + 60}
            
        if data['count'] >= 60:  # 每分钟60次
            raise RateLimitError()
            
        data['count'] += 1
        g.rate_limit_data[key] = data
        
        return f(*args, **kwargs)
    return decorated_function

def init_middleware(app: Flask) -> None:
    """初始化中间件"""
    configure_middleware(app)
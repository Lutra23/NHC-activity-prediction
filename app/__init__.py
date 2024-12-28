from flask import Flask
from flask_cors import CORS
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app.config import get_config

# 初始化扩展
cache = Cache()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

def create_app(config_name=None):
    """创建Flask应用实例"""
    app = Flask(__name__)
    
    # 加载配置
    config = get_config(config_name)
    app.config.from_object(config)
    config.init_app(app)
    
    # 初始化扩展
    CORS(app)
    cache.init_app(app)
    limiter.init_app(app)
    
    # 注册蓝图
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)
    
    # 注册错误处理
    register_error_handlers(app)
    
    return app

def register_error_handlers(app):
    """注册错误处理器"""
    @app.errorhandler(404)
    def not_found_error(error):
        return {
            'error': '找不到请求的资源'
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {
            'error': '服务器内部错误'
        }, 500
    
    @app.errorhandler(429)
    def ratelimit_handler(error):
        return {
            'error': '请求过于频繁，请稍后再试'
        }, 429
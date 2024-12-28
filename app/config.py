import os
import logging.handlers
from pathlib import Path
from typing import Dict, Any
from datetime import timedelta

# XTB 相关配置
XTB_HOME = Path(__file__).parent.parent / "xtb-6.7.1"
XTB_PATH = XTB_HOME / "bin" / "xtb.exe"
XTB_SHARE = XTB_HOME / "share" / "xtb"

# 打印XTB路径信息
print(f"\nXTB配置信息:")
print(f"XTB_HOME: {XTB_HOME}")
print(f"XTB_PATH: {XTB_PATH}")
print(f"XTB_SHARE: {XTB_SHARE}")

# 检查路径是否存在
if not XTB_HOME.exists():
    print(f"警告: XTB_HOME路径不存在: {XTB_HOME}")
if not XTB_PATH.exists():
    print(f"警告: XTB可执行文件不存在: {XTB_PATH}")
if not XTB_SHARE.exists():
    print(f"警告: XTB共享目录不存在: {XTB_SHARE}")

# 设置环境变量
os.environ["XTBHOME"] = str(XTB_HOME)
os.environ["XTBPATH"] = f"{XTB_SHARE};{os.path.expanduser('~')}"
os.environ["PATH"] = f"{os.environ['PATH']};{XTB_HOME / 'bin'}"

# 打印环境变量
print(f"\n环境变量设置:")
print(f"XTBHOME: {os.environ.get('XTBHOME')}")
print(f"XTBPATH: {os.environ.get('XTBPATH')}")

class Config:
    """基础配置类"""
    # 应用根目录
    BASE_DIR = Path(__file__).parent.parent
    
    # 模型文件路径
    MODEL_DIR = BASE_DIR / 'models'
    BOND_ORDER_MODEL = MODEL_DIR / 'rf_bond_order_predictor.joblib'
    AROMATICITY_MODEL = MODEL_DIR / 'rf_aromaticity_predictor.joblib'
    FEATURE_SCALER = MODEL_DIR / 'feature_scaler.joblib'
    
    # 确保必要目录存在
    MODEL_DIR.mkdir(exist_ok=True)
    
    # 日志配置
    LOG_DIR = BASE_DIR / 'logs'
    LOG_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOG_DIR / 'app.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = logging.INFO
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # 缓存配置
    CACHE_DIR = BASE_DIR / 'cache'
    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_TYPE = 'filesystem'
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_THRESHOLD = 1000
    CACHE_DIR = str(CACHE_DIR)
    
    # 限流配置
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = '60/minute'
    RATELIMIT_HEADERS_ENABLED = True
    
    # 安全配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-please-change-in-production')
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS配置
    CORS_ENABLED = True
    CORS_ORIGINS = ['*']
    CORS_METHODS = ['GET', 'POST', 'OPTIONS']
    CORS_ALLOW_HEADERS = ['Content-Type']
    CORS_MAX_AGE = 600
    
    # API配置
    API_TITLE = '化学键级预测API'
    API_VERSION = 'v1'
    API_PREFIX = '/api/v1'
    OPENAPI_VERSION = '3.0.2'
    OPENAPI_URL_PREFIX = '/docs'
    OPENAPI_SWAGGER_UI_PATH = '/swagger'
    OPENAPI_SWAGGER_UI_URL = 'https://cdn.jsdelivr.net/npm/swagger-ui-dist/'
    
    # 模型配置
    MODEL_BATCH_SIZE = 32
    MODEL_TIMEOUT = 30  # 秒
    MODEL_MAX_RETRIES = 3
    MODEL_RETRY_DELAY = 1  # 秒
    
    # 特征提取配置
    FEATURE_MAX_RADIUS = 3
    FEATURE_USE_CHIRALITY = True
    FEATURE_USE_FEATURES = True
    
    # 其他配置
    DEBUG = False
    TESTING = False
    PROPAGATE_EXCEPTIONS = True
    JSON_SORT_KEYS = False
    JSON_AS_ASCII = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    @staticmethod
    def init_app(app):
        """初始化应用配置
        
        Args:
            app: Flask应用实例
        """
        # 配置日志
        handler = logging.handlers.RotatingFileHandler(
            filename=str(Config.LOG_FILE),
            maxBytes=Config.LOG_MAX_BYTES,
            backupCount=Config.LOG_BACKUP_COUNT
        )
        handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        handler.setLevel(Config.LOG_LEVEL)
        
        app.logger.addHandler(handler)
        app.logger.setLevel(Config.LOG_LEVEL)
        
        # 其他初始化...
    
class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
    SESSION_COOKIE_SECURE = False
    CORS_ORIGINS = ['http://localhost:*']
    
class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
    CACHE_TYPE = 'simple'
    SESSION_COOKIE_SECURE = False
    CORS_ORIGINS = ['http://localhost:*']
    MODEL_PATH = 'path/to/test/models'
    RATELIMIT_STRATEGY = 'fixed-window'
    
class ProductionConfig(Config):
    """生产环境配置"""
    LOG_LEVEL = logging.WARNING
    CORS_ORIGINS = ['https://*.example.com']
    TESTING = False
    CACHE_TYPE = 'redis'
    MODEL_PATH = 'path/to/prod/models'
    RATELIMIT_STRATEGY = 'fixed-window'
    
    # 覆盖敏感配置
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # 生产环境定配置
        import logging.handlers
        
        # 配置邮件错误通知
        mail_handler = logging.handlers.SMTPHandler(
            mailhost=os.getenv('MAIL_SERVER'),
            fromaddr=os.getenv('MAIL_SENDER'),
            toaddrs=[os.getenv('ADMIN_EMAIL')],
            subject='应用错误'
        )
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)

# 配置映射
config: Dict[str, Any] = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(env: str = None) -> Any:
    """获取配置类
    
    Args:
        env: 环境名称,默认从环境变量获取
        
    Returns:
        配置类
    """
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default']) 
from typing import Dict, Any, Optional
from flask import jsonify
import logging

logger = logging.getLogger(__name__)

class APIError(Exception):
    """自定义API异常基类"""
    def __init__(self, 
                 message: str, 
                 code: int = 400, 
                 status: str = "error", 
                 details: Optional[Dict[str, Any]] = None,
                 log_level: str = "error"):
        self.message = message
        self.code = code
        self.status = status
        self.details = details or {}
        
        # 记录错误
        log_func = getattr(logger, log_level)
        log_func(f"{status}: {message} - {details}")
        
        super().__init__(self.message)

class ValidationError(APIError):
    """输入验证错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message, 
            code=400, 
            status="validation_error", 
            details=details,
            log_level="warning"
        )

class ModelError(APIError):
    """模型预测错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message, 
            code=500, 
            status="model_error", 
            details=details,
            log_level="error"
        )

class MoleculeError(APIError):
    """分子处理错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message, 
            code=400, 
            status="molecule_error", 
            details=details,
            log_level="warning"
        )

class CarbeneError(APIError):
    """卡宾结构错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message, 
            code=400, 
            status="carbene_error", 
            details=details,
            log_level="warning"
        )

class RateLimitError(APIError):
    """速率限制错误"""
    def __init__(self, message: str = "请求过于频繁,请稍后再试", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message, 
            code=429, 
            status="rate_limit_error", 
            details=details,
            log_level="warning"
        )

class CacheError(APIError):
    """缓存错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message, 
            code=500, 
            status="cache_error", 
            details=details,
            log_level="error"
        )

def init_error_handlers(app):
    """初始化错误处理器
    
    Args:
        app: Flask应用实例
    """
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """处理API错误"""
        response = {
            "status": error.status,
            "message": error.message,
            "details": error.details
        }
        return jsonify(response), error.code

    @app.errorhandler(404)
    def handle_404(error):
        """处理404错误"""
        response = {
            "status": "error",
            "message": "资源未找到",
            "details": {
                "path": error.description,
                "method": error.name
            }
        }
        logger.warning(f"404错误: {error.description}")
        return jsonify(response), 404

    @app.errorhandler(405)
    def handle_405(error):
        """处理405错误"""
        response = {
            "status": "error",
            "message": "不支持的请求方法",
            "details": {
                "path": error.description,
                "method": error.name,
                "allowed_methods": error.valid_methods
            }
        }
        logger.warning(f"405错误: {error.description}")
        return jsonify(response), 405

    @app.errorhandler(500)
    def handle_500(error):
        """处理500错误"""
        response = {
            "status": "error",
            "message": "服务器内部错误",
            "details": {
                "error": str(error),
                "type": error.__class__.__name__
            }
        }
        logger.error(f"500错误: {str(error)}")
        return jsonify(response), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """处理未预期的错误"""
        response = {
            "status": "error",
            "message": "发生未预期的错误",
            "details": {
                "error": str(error),
                "type": error.__class__.__name__
            }
        }
        logger.exception("未预期的错误")
        return jsonify(response), 500 
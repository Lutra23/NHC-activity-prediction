<<<<<<< HEAD
# 化学键级预测系统

基于机器学习的化学键级预测系统，用于预测分子中的化学键级和芳香性。

## 功能特点

- 支持单分子键级预测
- 支持批量分子处理
- 提供REST API接口
- 支持多种分子输入格式
- 高性能缓存机制

## 安装说明

1. 克隆项目
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 启动服务
```bash
python wsgi.py
```

### API使用示例

1. 单分子预测
```python
import requests

response = requests.post('http://localhost:5000/api/v1/predict/carbene-bond', 
    json={
        'smiles': 'CC(C)[C]',
        'carbene_idx': 3
    }
)
print(response.json())
```

2. 批量预测
```python
response = requests.post('http://localhost:5000/api/v1/predict/carbene-bond/batch',
    json={
        'molecules': [
            {'smiles': 'CC(C)[C]', 'carbene_idx': 3},
            {'smiles': 'CC(C)(C)[C]', 'carbene_idx': 4}
        ]
    }
)
print(response.json())
```

## 配置说明

主要配置文件位于 `app/config.py`，包含：
- 模型配置
- 缓存设置
- API限流
- 日志配置

## 开发说明

### 运行测试
```bash
pytest
```

### 目录结构
```
app/
├── models/     # 预测模型
├── routes/     # API路由
├── utils/      # 工具函数
└── config.py   # 配置文件
```
=======
# NHC-activity-prediction
>>>>>>> 9fa1735ab1a951665434f902937b97c30e92c717

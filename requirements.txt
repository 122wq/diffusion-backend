# 1. 使用官方 Python 3.10 slim 基础镜像
FROM python:3.10-slim

WORKDIR /app

# 2. 安装 ONNXRuntime 和 SciPy 所需的底层 C/C++ 运行库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 升级 pip 工具链
RUN pip install --no-cache-dir --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

# 4. 复制依赖清单并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 5. 复制当前目录下的所有代码及模型文件 (包含 main.py 和 onnx_diffusion.onnx)
COPY . .

# 6. 暴露 80 端口
EXPOSE 80

# 7. 启动 uvicorn 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
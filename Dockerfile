FROM python:3.9-slim

WORKDIR /app

# 1. 安装基础编译环境
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. 核心解决点：先升级 pip、setuptools 和 wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

COPY requirements.txt .

# 3. 安装依赖（换用更稳定的阿里云源）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

COPY . .

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
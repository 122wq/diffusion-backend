# 1. 升级为 Python 3.10 镜像，解决大部分 >=3.10 的依赖版本限制
FROM python:3.10-slim

WORKDIR /app

# 2. 安装必要的轻量编译基础环境（防止部分需要 C 扩展的包报错）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

COPY requirements.txt .

# 4. 正常安装依赖（移除 --only-binary=:all: 限制，允许像 docopt 这样的纯源码包安装）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

COPY . .

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
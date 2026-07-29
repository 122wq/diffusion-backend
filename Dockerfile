FROM python:3.9-slim

WORKDIR /app

# 先升级基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

COPY requirements.txt .

# 关键：加上 --only-binary=:all: 强制下载编译好的 .whl 包，避免本地耗时编译！
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

COPY . .

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
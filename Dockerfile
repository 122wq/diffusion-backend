FROM python:3.9-slim

WORKDIR /app

# 安装基础构建环境
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 80

CMD ["unicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
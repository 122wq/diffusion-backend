FROM python:3.9-slim

WORKDIR /app

# 1. 安装基础编译环境（解决需要 C/C++ 编译依赖的包）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. 先升级 pip（推荐）
RUN pip install --no-cache-dir --upgrade pip

# 3. 安装依赖（加入了 trusted-host 参数）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

# 4. 复制代码
COPY . .

EXPOSE 80

# 5. 修正了 uvicorn 的拼写
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
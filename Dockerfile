FROM python:3.11-slim

# LightGBM 依赖 OpenMP 运行时；slim 镜像里没有
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独一层，改代码时不必重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces 默认 7860；Render / Railway 会注入 PORT
ENV PORT=7860
EXPOSE 7860

# 单进程多线程：107k 行数据和模型在内存里只有一份。
# 换成多 worker 的话每个 worker 都会各存一份，free tier 的内存扛不住。
CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --threads=4 app:app"]

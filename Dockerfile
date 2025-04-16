# 使用官方 Python 基础镜像
FROM python:3.9-slim

# 安装系统依赖 + 中文字体
RUN apt-get update && apt-get install -y \
    fonts-wqy-zenhei \   # 文泉驿正黑字体（解决中文显示）
    fonts-wqy-microhei \  # 可选：文泉驿微米黑
    && rm -rf /var/lib/apt/lists/*

# 更新字体缓存（关键步骤！）
RUN fc-cache -fv

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码到容器
COPY . .

# 启动命令（运行你的 Streamlit 应用）
CMD ["streamlit", "run", "your_app.py"]
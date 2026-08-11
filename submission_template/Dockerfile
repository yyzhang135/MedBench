# 参赛者可以选择自己所需的的基础镜像，例如 PyTorch 官方镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装必要的 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 将所有推理代码拷贝进容器
COPY . .

# 规定容器启动时的默认执行命令，参数将由评测后台动态传入
ENTRYPOINT ["python", "run_inference.py"]

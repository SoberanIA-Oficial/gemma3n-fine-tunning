FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install CUDA keyring and toolkit
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-9

# Install all system dependencies, including git, in one RUN command
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install latest transformers for Gemma 3N
RUN pip install --no-deps git+https://github.com/huggingface/transformers.git
#RUN pip install --no-deps --upgrade timm 
#RUN pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
# Install timm for Gemma 3N
RUN pip install --no-deps --upgrade timm xformers==0.0.29.post3
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install deepspeed
COPY . /app/

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-12.9/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

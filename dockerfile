FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

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
# Install timm for Gemma 3N
RUN pip install --no-deps --upgrade timm xformers==0.0.29.post3
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

COPY . /app/

# Gemma 3N Fine-tuning for ENEM Questions 🇧🇷

This project implements fine-tuning of Google's Gemma 3N models on Brazilian ENEM (Exame Nacional do Ensino Médio) exam questions using Unsloth for efficient training.

## 📋 Overview

This repository contains:
- Fine-tuning scripts for Gemma 3N models on ENEM dataset
- Data processing pipelines for ENEM questions with reasoning/explanation of correct question
- Docker setup for easy deployment and inference
- Baseline inference notebooks for model evaluation
- LLaMA.cpp integration for efficient model serving

## 🚀 Features

- **Efficient Fine-tuning**: Uses Unsloth library for fast and memory-efficient training
- **Reasoning Enhancement**: Incorporates detailed reasoning for ENEM questions
- **Multi-year Support**: Processes ENEM data from 2022, 2023, and 2024
- **LoRA Training**: Implements parameter-efficient fine-tuning with LoRA adapters
- **Docker Support**: Complete containerization for development and deployment
- **Model Serving**: LLaMA.cpp integration for high-performance inference

## 📁 Project Structure

```
├── BASELINE-inference.ipynb          # Baseline model inference and evaluation
├── Unsloth-Enem-finetune-lora-reason.ipynb  # Main fine-tuning notebook with reasoning
├── process_dataset.ipynb            # Data preprocessing and preparation
├── enem.py                          # Core training script
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Main Docker setup
├── docker-compose-llamacpp.yml      # LLaMA.cpp server setup
├── dockerfile                       # Docker image definition
├── data/                           # ENEM datasets (processed)
└── models/                         # Fine-tuned models and GGUF files

```

## 🛠️ Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/SoberanIA-Oficial/gemma3n-fine-tunning.git
cd gemma3n-fine-tunning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup (Recomended)

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. For LLaMA.cpp server:
```bash
docker-compose -f docker-compose-llamacpp.yml up
```

## 📚 Dataset

The project uses the ENEM dataset from Maritaca AI:
- **Source**: `maritaca-ai/enem` on Hugging Face
- **Years**: 2022, 2023, 2024
- **Enhanced**: With custom reasoning explanations for better model understanding (Generated from another LLM )
- **Processing**: Automated filtering of cancelled questions and data formatting

## 🔧 Usage

### Data Processing

Run the data processing notebook to prepare ENEM questions:
```bash
jupyter notebook process_dataset.ipynb
```

### Fine-tuning

**Interactive Training**: Use the main fine-tuning notebook:
```bash
jupyter notebook Unsloth-Enem-finetune-lora-reason.ipynb
```

```

### Inference

1. **Baseline Evaluation**:
```bash
jupyter notebook BASELINE-inference.ipynb
```

2. **Server Inference**: Start the LLaMA.cpp server and make requests:
```bash
curl -X POST http://localhost:8001/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your ENEM question here", "max_tokens": 512}'
```

## 🏗️ Model Architecture

- **Base Model**: Gemma 3N (E2B/E4B variants)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for memory efficiency
- **Context Length**: Up to 2048 tokens
- **Training Strategy**: Supervised Fine-Tuning (SFT) with reasoning

## 📊 Training Configuration (configurable)

- **Batch Size**: 8 per device
- **Learning Rate**: 2e-4
- **Epochs**: 1 
- **Optimizer**: AdamW with weight decay
- **Memory Optimization**: Gradient checkpointing and 4-bit quantization

## 🎯 Model Performance

The fine-tuned models are designed to:
- Answer ENEM multiple-choice questions accurately
- Provide detailed reasoning for each answer
- Understand Portuguese language nuances in academic contexts
- Handle various subject areas covered in ENEM

## 📁 Output Models

Generated models are saved in multiple formats:
- **Unsloth Format**: For continued training and adaptation
- **GGUF Format**: For efficient inference with LLaMA.cpp
- **Hugging Face Format**: For easy deployment and sharing

## 🚦 GPU Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 16GB+ VRAM
- **Multi-GPU**: Supported via Docker configuration

## 📄 License

This project is part of the Google Gemma 3N Hackathon and follows the respective licensing terms.

---

🇧🇷 **Desenvolvido no Brasil** | Built in Brazil by SoberanIA
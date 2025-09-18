FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /app
RUN apt-get update && apt-get install -y git curl
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers datasets accelerate peft bitsandbytes trl langdetect ollama sentencepiece "protobuf<5"

CMD ["bash"]
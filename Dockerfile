FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y git curl libpng-dev libjpeg-dev
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
RUN pip install transformers datasets accelerate peft bitsandbytes trl langdetect ollama sentencepiece "protobuf<5"

CMD ["bash"]
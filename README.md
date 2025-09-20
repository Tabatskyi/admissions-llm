## What you need:
- Docker Desktop with WSL2 backend
- NVIDIA GPU with CUDA support and latest drivers
- Hugging Face account with access to [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model and [Hugging Face token](https://huggingface.co/settings/tokens) in the *.env* file with `HUGGING_FACE_HUB_TOKEN` variable
- Dataset in JSONL format with `prompt` and `completion` fields, for example [this one](admissions_data.jsonl).
- *ollama* package for python: `pip install ollama`

## How this works:
0. Build training image: `docker compose --profile finetune build` (one time)
1. Run training container: `docker compose --profile finetune up --force-recreate`, if everything ok you will receive message and `fine-tuned-mistral` directory will be filled
2. Run AI container: `docker compose --profile ollama up`
3. Run client: `python query_bot.py`
4. Start messaging

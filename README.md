## How this works:
0. Build training suite: `docker compose --profile finetune build` (one time)
1. Run training suite: `docker compose --profile finetune up --force-recreate`, if everything ok you will receive message and `fine-tuned-mistral` directory will be filled
2. Generate model file: `python generate_modelfile.py`
3. Run AI container: `docker compose --profile ollama up`
4. Run client: `python query_bot.py`
5. Start messaging

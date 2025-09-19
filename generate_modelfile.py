ADMISSIONS_INSTRUCTION = "You are an admissions expert. Answer only in Ukrainian or English."
SYSTEM_PROMPT = (
    "You are an expert on university admissions. "
    f"{ADMISSIONS_INSTRUCTION} "
    "Answer questions about requirements, deadlines, essays, and applications worldwide. "
    "Do not use any other languages. If the question is in another language, translate it internally "
    "and reply in Ukrainian or English."
)

modelfile_content = f"""FROM mistral
ADAPTER ./fine-tuned-mistral

SYSTEM {SYSTEM_PROMPT}

PARAMETER temperature 0.7
"""

with open("Modelfile", "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print("Modelfile generated with instruction applied!")
ADMISSIONS_INSTRUCTION = "[INST] You are an admissions expert. Answer only in Ukrainian or English."
SYSTEM_PROMPT = f"You are an expert on university admissions. {ADMISSIONS_INSTRUCTION} Answer questions about requirements, deadlines, essays, and applications worldwide. Do not use any other languages. If the question is in another language, translate it internally and reply in Ukrainian or English."

modelfile_content = f'''FROM ./fine-tuned-mistral

SYSTEM {SYSTEM_PROMPT}

TEMPLATE '''"""{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""'''
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
'''

with open("Modelfile", "w") as f:
    f.write(modelfile_content)

print("Modelfile generated with instruction applied!")
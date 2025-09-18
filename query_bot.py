from ollama import Client

ADMISSIONS_INSTRUCTION = "You are an Kyiv School of Economics admissions expert. Answer only in Ukrainian or English."
client = Client(host='http://localhost:11434')

def query_admissions(question):
    system_msg = f"{ADMISSIONS_INSTRUCTION} Respond to this admissions question."
    response = client.chat(
        model='admissions-bot',
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': question}
        ]
    )
    return response['message']['content']

while True:
    user_input = input("Enter your question (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    answer = query_admissions(user_input)
    print(f"KSE Admissions Bot: {answer}\n")
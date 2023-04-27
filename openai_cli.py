import openpyxl
import requests
import json
from colorama import Fore, Style, init

init(autoreset=True)
def read_first_cell():
    file_path="../openaikey.xlsx" ## use your own key
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    first_cell = sheet.cell(row=1, column=1)
    return first_cell.value

def generate_chat_completion(first_cell_value, messages, model="gpt-3.5-turbo", temperature=1, max_tokens=None):
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {first_cell_value}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def main():
    first_cell_value = read_first_cell()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        response_text = generate_chat_completion(first_cell_value, messages)
        print("\nAssistant:", end=" ")
        print(Fore.WHITE + response_text)
        messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()


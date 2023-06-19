import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from colorama import Fore, Back, Style, init
import time
import json
import datasets

id = 0
data_list = []

while True:
    input_text = input('user ')
    inputs = tokenizer(input_text , return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(Fore.YELLOW + 'CALM: ' + output)
    id +=1
    print(f"id: {id}")

    data = {"id": id, "input_text": input_text, "output":output , "timestamp":time.time()}
    data_list.append(data)

    with open('data.json', 'w') as file:
        json.dump(data_list, file, ensure_ascii=False)
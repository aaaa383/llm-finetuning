import os
import datetime
import gradio as gr
from gradio.mix import Parallel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

model_path = "rinna/japanese-gpt-neox-3.6b"
# cyberagent/open-calm-7b
base_llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# torch_dtype=torch.float16
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)
model1 = PeftModel.from_pretrained(base_llm, '../model/lora-rinna-3.6b-sakura_dataset')
# torch_dtype=torch.float16
# model1.half()

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""### 指示:
            {data_point["instruction"]}

            ### 入力:
            {data_point["input"]}

            ### 回答:
        """
    else:
        result = f"""### 指示:
            {data_point["instruction"]}

            ### 回答:
        """

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result

# テキスト生成関数の定義
def generate(instruction,input=None,maxTokens=256):
    # 推論
    prompt = generate_prompt({'instruction':instruction,'input':input})
    input_ids = tokenizer(prompt, 
        return_tensors="pt", 
        truncation=True, 
        add_special_tokens=False).input_ids.cuda()

    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=maxTokens, 
        do_sample=True,
        temperature=0.7, 
        top_p=0.75, 
        top_k=40,         
        no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()
    print(tokenizer.decode(outputs))

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "### 回答:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc+len(sentinel):]
            print(result.replace("<NL>", "\n"))  # <NL>→改行
            result = result.replace("<NL>", "\n")
        else:
            print('Warning: Expected prompt template to be emitted.  Ignoring output.')
    else:
        print('Warning: no <eos> detected ignoring output')

    return result

# def chat(model, instruction, temp, conversation=''):
    
#     # 温度パラメータの入力制限
#     if temp < 0.1:
#         temp = 0.1
#     elif temp > 1:
#         temp = 1
    
#     input_prompt = {
#         'instruction': instruction,
#         'input': conversation
#     }
    
#     prompt = ''
#     if input_prompt['input'] == '':
    
#         prompt = PROMPT_DICT['prompt_no_input'].format_map(input_prompt)
#     else:
#         prompt = PROMPT_DICT['prompt_input'].format_map(input_prompt)

    
#     inputs = tokenizer(
#         prompt,
#         return_tensors='pt'
#     ).to(model.device)
    
#     with torch.no_grad():
#         tokens = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             do_sample=True,
#             temperature=temp,
#             top_p=0.9,
#             repetition_penalty=1.05,
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     output = tokenizer.decode(tokens[0], skip_special_tokens=True)
#     output = output.split('\n\n')[-1].split(':')[-1]
    
#     return output

def fn1(instruction):
    return generate(instruction,input=None,maxTokens=256)


examples = [
    ["日本の観光名所を3つ挙げて。"],
    ["データサイエンティストに必要なスキルを5つ挙げて。"],
    ["3+2は？"]
]

demo1= gr.Interface(
    fn1,
    [
        gr.inputs.Textbox(lines=5, label="入力テキスト"),
        # gr.inputs.Number(default=0.3, label='Temperature(0.1 ≦ temp ≦ 1 の範囲で入力してください\n小さい値ほど指示に忠実、大きい値ほど多様な表現を出力します。)'),
    ],
    gr.outputs.Textbox(),
)

demo = gr.Parallel(
    demo1,
    examples=examples,
    title="CyberAgent rinna-lora-instruction-tuning",
    description="""
        LoRAのチューニング方法
        """,)

# share=Trueで外部に公開される。
demo.launch(server_name="0.0.0.0", share=True)

# def greet(name):
#     return "Hello " + name + "!!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# # servernameをしてしてあげないと、外部から見えない。
# demo.launch(server_name="0.0.0.0", share=True)
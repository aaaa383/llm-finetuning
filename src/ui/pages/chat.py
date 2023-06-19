import os
import json
import requests
import streamlit as st
from streamlit_chat import message
import openai

# openAI版
openai.api_key = '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'

# ローカルでモデルを読み込む場合(Macで動かない)
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# with st.spinner('モデル読み込み中です、少々お待ちください'):
#   model_path = "cyberagent/open-calm-7b"
#   base_llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
#   # torch_dtype=torch.float16
#   tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)
#   model1 = PeftModel.from_pretrained(base_llm, 'ponponnsan/LoRAInstruction')
#   # torch_dtype=torch.float16
#   # model1.half()
# st.success('モデル読み込みが完了しました!')


# PROMPT_DICT = {
#     "prompt_input": (
#         "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
#         "要求を適切に満たす応答を書きなさい。\n\n"
#         "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
#     ),
#     "prompt_no_input": (
#         "以下は、タスクを説明する指示です。"
#         "要求を適切に満たす応答を書きなさい。\n\n"
#         "### 指示:\n{instruction}\n\n### 応答:"
#     )
# }


# def chat(model, instruction, temp=0.7, conversation=''):
    
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


# def fn1(instruction):
#     return chat(model1, instruction)


# UIのセットアップ
st.title('OptiMathPro(チャット画面)')

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []
if "name" not in st.session_state:
    st.session_state.name = ""

with st.form("メッセージを入力して下さい"):
  # openai_api_key = st.text_area("APIキーを入力して下さい")
  user_message = st.text_area("メッセージを入力して下さい")
  st.session_state.name = st.text_input("名前を入力してください", value=st.session_state.name) # 名前を入力

  if user_message is not None:

    submitted = st.form_submit_button("送信")
    if submitted:
      # メッセージが空白でないかチェックします
      if user_message.strip() == '' or st.session_state.name.strip() == '':
        st.warning("送信ボタンを押す前にメッセージと名前を入力して下さい")
      else:

        # チャット送信
        # answer = fn1(user_message)

        print('user_message',user_message)

        response = openai.Completion.create(
            engine='davinci:ft-personal-2023-06-13-05-16-24',
            prompt=user_message,
            max_tokens=100,
            stop='\n')
        answer = response['choices'][0]['text']
        print(user_message+response['choices'][0]['text'])

        # API Gatewayのパスを入れる
        api_endpoint = 'https://wtm35xxg9l.execute-api.ap-northeast-1.amazonaws.com/test/chat'

        # APIリクエストを送信します
        response = requests.post(api_endpoint, json={'message': user_message, 'name': st.session_state.name, 'answer': answer})
        
        # レスポンスを取得します
        # response_data = json.loads(response.text)
        # answer = response_data["body"]

        st.session_state.past.append(user_message)
        st.session_state.generated.append(answer)

        if st.session_state["generated"]:
          for i in range(len(st.session_state.generated)):
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
            message(st.session_state.generated[i], key=str(i))

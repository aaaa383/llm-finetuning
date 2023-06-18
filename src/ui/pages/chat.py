import os
import json
import requests
import streamlit as st
from streamlit_chat import message

# UIのセットアップ
st.title('OptiMathPro(チャット画面)')

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []
if "name" not in st.session_state:
    st.session_state.name = ""

with st.form("メッセージを入力して下さい"):
  user_message = st.text_area("メッセージを入力して下さい")
  st.session_state.name = st.text_input("名前を入力してください", value=st.session_state.name) # 名前を入力

  if user_message is not None:

    submitted = st.form_submit_button("送信")
    if submitted:
      # メッセージが空白でないかチェックします
      if user_message.strip() == '' or st.session_state.name.strip() == '':
        st.warning("送信ボタンを押す前にメッセージと名前を入力して下さい")
      else:

        # API Gatewayのパスを入れる
        api_endpoint = 'https://wtm35xxg9l.execute-api.ap-northeast-1.amazonaws.com/test/chat'

        # APIリクエストを送信します
        response = requests.post(api_endpoint, json={'message': user_message, 'name': st.session_state.name})
        
        # レスポンスを取得します
        response_data = json.loads(response.text)
        answer = response_data["body"]

        st.session_state.past.append(user_message)
        st.session_state.generated.append(answer)

        if st.session_state["generated"]:
          for i in range(len(st.session_state.generated)):
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
            message(st.session_state.generated[i], key=str(i))

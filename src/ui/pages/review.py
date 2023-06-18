import os
import json
import requests
import pandas as pd
import streamlit as st

# UIのセットアップ
st.title('OptiMathPro(レビュー画面)')

if "name" not in st.session_state:
    st.session_state.name = ""
if "review" not in st.session_state:
    st.session_state.review = ""
if "data" not in st.session_state:
    st.session_state.data = {}
if "selected_uuid" not in st.session_state:
    st.session_state.selected_uuid = ""

# 名前入力
st.session_state.name = st.text_input("ユーザー名を入力してください", value=st.session_state.name)

if st.session_state.name:
    # GETリクエストでデータを取得
    api_endpoint = 'https://wtm35xxg9l.execute-api.ap-northeast-1.amazonaws.com/test/chat'  
    response = requests.get(api_endpoint, params={'name': st.session_state.name})

    if response.status_code == 200:
        # レスポンスを取得し、表示します
        st.session_state.data = json.loads(response.text)
        df = pd.DataFrame(st.session_state.data)  # Convert the JSON to pandas DataFrame
        if len(df) == 0:
            st.write("該当するデータが存在しません。")
        else:
            df = df[['uuid', 'name', 'userInput','response', 'fix_response']]  # Select only the required columns
            st.table(df)  # Display the DataFrame as a table

            # If there are any uuids and a uuid hasn't been selected yet, select the first one
            if len(df['uuid'].tolist()) > 0 and st.session_state.selected_uuid == "":
                st.session_state.selected_uuid = df['uuid'].tolist()[0]

            # Create a select box with the uuids
            uuid = st.selectbox("UUIDを選択してください", options=df['uuid'].tolist(), key='selected_uuid')

            # Display the selected row
            selected_row = df[df['uuid'] == st.session_state.selected_uuid]
            st.write(selected_row)

            # Review text box
            st.session_state.review = st.text_input("responseの修正内容を記入してください", value=st.session_state.review)

            # Submission button
            if st.button("提出"):
                put_url = api_endpoint + '/' + uuid
                body = {'uuid': uuid, 'fix_response': str(st.session_state.review)}

                # リクエストの送信
                response = requests.put(put_url, data=json.dumps(body))

                print('response',response)

                if response.status_code == 200:
                    st.success("修正内容が提出されました。")
                else:
                    st.exception("修正内容の反映に失敗しました。")
           
    else:
        st.write("Error: ", response.status_code)

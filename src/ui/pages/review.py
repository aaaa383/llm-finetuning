import os
import json
import requests
import pandas as pd
import streamlit as st

# UIのセットアップ
st.title('OptiMathPro(レビュー画面)')

# 名前入力
name = st.text_input("ユーザー名を入力してください")

if name:
    # GETリクエストでデータを取得
    api_endpoint = 'https://wtm35xxg9l.execute-api.ap-northeast-1.amazonaws.com/test/chat'  
    response = requests.get(api_endpoint, params={'name': name})

    if response.status_code == 200:
        # レスポンスを取得し、表示します
        data_ = json.loads(response.text)
        df = pd.DataFrame(data_)  # Convert the JSON to pandas DataFrame
        if len(df) == 0:
            st.write("該当するデータが存在しません。")
        else:
            df = df[
                [
                    'uuid', 
                    'name', 
                    'userInput',
                    'response', 
                    'fix_response'
                ]
                ].rename(
                    columns={
                        'name': 'ユーザー名',
                        'userInput': '入力文',
                        'response': '出力文',
                        'fix_response': '修正内容'
                    }
                )
            st.table(df)  # Display the DataFrame as a table

            # If there are any uuids, select the first one by default
            if len(df['uuid'].tolist()) > 0:
                selected_uuid = df['uuid'].tolist()[0]

                # Create a select box with the uuids
                uuid = st.selectbox("UUIDを選択してください", options=df['uuid'].tolist(), index=0)

                # Display the selected row
                selected_row = df[df['uuid'] == uuid]
                st.write(selected_row)

                # Review text box
                review = st.text_input("responseの修正内容を記入してください")

                # Submission button
                if st.button("提出"):
                    put_url = api_endpoint + '/' + uuid
                    body = {'fix_response': str(review)}

                    # リクエストの送信
                    response = requests.put(put_url, data=json.dumps(body))
                    st.success("修正内容が提出されました。")
                    # if response.status_code == 200:
                    #     st.success("修正内容が提出されました。")
                    # else:
                    #     st.exception("修正内容の反映に失敗しました。")
           
    else:
        st.write("Error: ", response.status_code)

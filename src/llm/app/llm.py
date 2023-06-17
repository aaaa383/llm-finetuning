import boto3
import os
import json
import uuid
import time

# モデルがデプロイされているSageMakerエンドポイント名を環境変数から取得
endpoint_name = os.environ['ENDPOINT_NAME']

# SageMakerランタイムとDynamoDBクライアントを初期化
runtime = boto3.Session().client(service_name='sagemaker-runtime',region_name='us-west-2')
dynamodb = boto3.resource('dynamodb')

# DynamoDBテーブル名を指定
table_name = 'llm-table'
table = dynamodb.Table(table_name)

def lambda_handler(event, context):
    # ユーザーからのリクエストをSageMakerエンドポイントに送信
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType='text/plain', 
                                       Body=event['body'])

    # レスポンスをJSON形式に変換
    result = json.loads(response['Body'].read().decode())

    # DynamoDBに保存
    table.put_item(
        Item={
            'uuid': str(uuid.uuid1()),
            'timestamp': str(time.time()),
            'userInput': event['body'],
            'response': str(result)
        }
    )
    
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }

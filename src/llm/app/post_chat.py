import boto3
import os
import json
import uuid
import time

# モデルがデプロイされているSageMakerエンドポイント名を環境変数から取得
# endpoint_name = os.environ['ENDPOINT_NAME']
# runtime = boto3.Session().client('sagemaker-runtime')

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ["table_name"])

def lambda_handler(event, context):
    
    # DynamoDBに保存
    table.put_item(
        Item={
            'uuid': str(uuid.uuid1()),
            'timestamp': str(time.time()),
            'name': event['name'],
            'userInput': event['message'],
            'response': event['answer'],
            'fix_response': ''
        }
    )
    
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : event['answer']
    }
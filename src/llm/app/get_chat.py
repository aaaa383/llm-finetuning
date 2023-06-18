import os
import json
import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

# DynamoDBオブジェクトを作成します
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ["table_name"])

def lambda_handler(event, context):
    
    print(event)
    
    # イベントからnameパラメータを取得します
    name = event['queryStringParameters']['name']

    try:
        # DynamoDBからnameに一致するデータを取得します
        response = table.scan(
            FilterExpression=Attr('name').eq(name)
        )
        items = response['Items']

        # 結果を返します
        return {
            'statusCode': 200,
            'body': json.dumps(items),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' 
            }
        }

    except ClientError as e:
        # エラーが発生した場合はエラーメッセージを返します
        return {
            'statusCode': 500,
            'body': json.dumps(str(e)),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  
            }
        }
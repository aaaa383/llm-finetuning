import os
import boto3
from botocore.exceptions import ClientError


# DynamoDBオブジェクトの作成
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ["table_name"])

def lambda_handler(event, context):
    
    print(event)

    # クエリパラメータからuuidを取得
    uuid = event['uuid']
    print(uuid)

    # リクエストボディからfix_responseを取得
    fix_response = event['fix_response']
    print(fix_response)

    # 項目の更新
    try:
        response = table.update_item(
            Key={'uuid': uuid},
            UpdateExpression='SET fix_response = :val1',
            ExpressionAttributeValues={
                ':val1': fix_response
            },
            ReturnValues="UPDATED_NEW"
        )
        
        return {
            'statusCode': 200,
            'body': response
        }
        
    except ClientError as e:
        print(e.response['Error']['Message'])
        return {
            'statusCode': 500,
            'body': e.response['Error']['Message']
        }

import boto3
import os
import json
import uuid
import time

# モデルがデプロイされているSageMakerエンドポイント名を環境変数から取得
endpoint_name = os.environ['ENDPOINT_NAME']
runtime = boto3.Session().client('sagemaker-runtime')

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ["table_name"])

def lambda_handler(event, context):
    
    print('event',event)
    
    
    PROMPT_DICT = {
         "prompt_input": (
              "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
              "要求を適切に満たす応答を書きなさい。\n\n"
              "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
             ),
         "prompt_no_input": (
             "以下は、タスクを説明する指示です。"
             "要求を適切に満たす応答を書きなさい。\n\n"
             "### 指示:\n{instruction}\n\n### 応答:"
         )
    }
    
    instruction = event['message']
    conversation = ""
    temp = 0.7  

    input_prompt = {
        'instruction': instruction,
        'input': conversation
    }

    prompt = ''
    if input_prompt['input'] == '':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(input_prompt)
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(input_prompt)

    # Use the prompt as the data for the inference
    data = prompt
    print("data",data)

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=data,
        ContentType='text/plain',    # Update the MIME type if necessary
        Accept='Accept'
    )
    

    # The response is an HTTP response whose body contains the result of your inference
    result = response['Body'].read().decode()
    
    print('result',result)

    # レスポンスをJSON形式に変換
    result = json.loads(response['Body'].read().decode())
    
    # DynamoDBに保存
    table.put_item(
        Item={
            'uuid': str(uuid.uuid1()),
            'timestamp': str(time.time()),
            'name': event['name'],
            'userInput': event['message'],
            'response': str(result),
            'fix_response': ''
        }
    )
    
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }
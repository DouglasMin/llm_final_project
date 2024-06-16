import streamlit as st
import json
import boto3
import logging


bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")



def get_response(modelId, prompt_data):
    
    print("Model {} ".format(modelId))
    print("Message {} ".format(prompt_data))
    # max_tokens = 4096
    # temperature = 0
    # top_p = 0.9
    output_text = "model not working lol"

    if 'amazon' in modelId:
        
        
        print("amazon selected")
        
        request = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": {
                "maxTokenCount": 30,
                "temperature": 0.2,
            },
        })
        
        # Invoke the model with the request.
        response = bedrock_runtime.invoke_model(modelId=modelId, body=request)
        
        # Decode the response body.
        model_response = json.loads(response["body"].read())
        output_text = model_response["results"][0]["outputText"]
        
        print("Model format: {}" .format(model_response))
        
        
    elif 'anthropic' in modelId:
        
        
        print("claude selected")
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_data}],
                    }
                ],
            }
        )
        response = bedrock_runtime.invoke_model(
            modelId=modelId,
            body=body,
        )
        response_body = json.loads(response.get("body").read())
        output_text = response_body["content"][0]["text"]
        
        
        
    elif 'ai21' in modelId:
        
        print("ai21 selected")
        request = json.dumps(
            {
                "prompt": prompt_data,
                "maxTokens": 200,
                "temperature": 0.2,
            }
        )
        
        response = bedrock_runtime.invoke_model(modelId=modelId, body=request)
        model_response = json.loads(response["body"].read())   
        
        print("AI21 Labs format")
        
        
        output_text = model_response["completions"][0]["data"]["text"]
        print(output_text)
        
    elif 'mistral' in modelId:
        prompt = f"<s>[INST] {prompt_data} [/INST]"
        request = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.5,
            }
        )
        response = bedrock_runtime.invoke_model(modelId=modelId, body=request)
        
        model_response = json.loads(response["body"].read())
        output_text = model_response["outputs"][0]["text"]
        

    elif 'meta' in modelId:
        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {prompt_data}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        
        request = json.dumps(
            {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
            })
        
        response = bedrock_runtime.invoke_model(modelId=modelId, body=request)
        model_response = json.loads(response["body"].read())
        output_text = model_response["generation"]
        
    else:
        print('Parameter model must be one of titan, claude, j2, or sd')
        return


    return output_text

import streamlit as st
import json
import boto3
import logging

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Bedrock runtime
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

st.title("Chatbot powered by Bedrock with multi-choice model")

# Initialize session state if there are no messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.token_count = 0

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.token_count = 0

# Define model options
MODEL_OPTIONS = {
    "Claude": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Llama": "meta.llama3-8b-instruct-v1:0",
    "AI21 Lab": "ai21.j2-mid-v1",
    "Amazon Titan": "amazon.titan-text-premier-v1:0",
    "Mistral": "mistral.mistral-large-2402-v1:0",
}

def get_rMatthewesponse(modelId, prompt_data):
    
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

# Sidebar
with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With DynamoDB Memory :brain:')
    st.button('Clear Screen', on_click=clear_screen)
    
    option = st.selectbox(
        "What model would you like to use?",
        options=list(MODEL_OPTIONS.keys()))
    st.write("You selected:", option)

if user_prompt := st.chat_input("What's Up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Prepare the conversation history for the model
    conversation_history = "\n".join(
        f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages
    )
    full_response = get_response(MODEL_OPTIONS[option], conversation_history)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.chat_message("assistant"):
        st.markdown(full_response)

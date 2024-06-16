import streamlit as st
import json
import boto3
# Bedrock runtime
bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
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
    
def get_response(prompt_data):
    input_data = {
        'input': {
            'text': prompt_data
        },
        'retrieveAndGenerateConfiguration': {
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': 'MK8YAP5J5O',
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
            },
            'type': 'KNOWLEDGE_BASE'
        }
    }
    response = bedrock_runtime.retrieve_and_generate(**input_data)
    output_text = response['output']['text']
    return output_text
# Sidebar
with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With DynamoDB Memory :brain:')
    st.button('Clear Screen', on_click=clear_screen)
if user_prompt := st.chat_input("What's Up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    # Prepare the conversation history for the model
    conversation_history = "\n".join(
        f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages
    )
    full_response = get_response(conversation_history)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)ㅊ
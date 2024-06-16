import streamlit as st
import json
import boto3
from all_model_no_streaming import get_response as gr

# Bedrock runtime
MODEL_OPTIONS = {
    "Ordinary": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Knowledge_Based": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
}

ORDINARY_MODEL_OPTION = {
    "Claude": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Llama": "meta.llama3-8b-instruct-v1:0",
    "AI21 Lab": "ai21.j2-mid-v1",
    "Amazon Titan": "amazon.titan-text-premier-v1:0",
    "Mistral": "mistral.mistral-large-2402-v1:0",
}

bedrock_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name='us-east-1')
bedrock_runtime2 = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

st.title("Knowledge Based 챗봇: 물어보살 :innocent: ")

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

def get_response_kb(prompt_data):
    input_data = {
        'input': {
            'text': prompt_data
        },
        'retrieveAndGenerateConfiguration': {
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': 'MK8YAP5J5O',
                'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'
            },
            'type': 'KNOWLEDGE_BASE',
        }
    }
    response = bedrock_runtime.retrieve_and_generate(**input_data)
    output_text = response['output']['text']
    return output_text

# Sidebar
with st.sidebar:
    st.title('사내 문서에 대해 궁금한 것을 물어보세요! :robot_face: :computer: ')
    st.subheader('기억도 할 수 있어요! :brain:')
    st.button('Clear Screen', on_click=clear_screen, key='clear_screen_button')

    option1 = st.selectbox(
        "옵션",
        options=list(MODEL_OPTIONS.keys()))
    st.write("모드:", option1)

    if option1 == "Ordinary":
        option2 = st.selectbox(
            "FM 모델을 고르시오",
            options=list(ORDINARY_MODEL_OPTION.keys()), key='ordinary_model_select')
        st.write("선택된 FM 모델:", option2)
    else:
        option2 = None  # Ensure option2 is defined even if not used

if user_prompt := st.chat_input("What's Up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    # Prepare the conversation history for the model
    conversation_history = "\n".join(
        f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages
    )

    if option1 == "Ordinary" and option2:
        full_response = gr(ORDINARY_MODEL_OPTION[option2], conversation_history)
    elif option1 == "Knowledge_Based":
        full_response = get_response_kb(conversation_history)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)

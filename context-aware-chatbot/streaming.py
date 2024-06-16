import streamlit as st
import json
import boto3

# Bedrock runtime
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

st.title("Chatbot powered by Bedrock with multi-choice model")

# Initialize session state if there are no messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.token_count = 0

# Function to estimate the number of tokens in a message
def count_tokens(text):
    return len(text.split())

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.token_count = 0

def chunk_handler(chunk):
    text = ""
    chunk_type = chunk.get("type")
    if chunk_type == "message_start":
        text = ""
    elif chunk_type == "content_block_start":
        text = chunk["content_block"]["text"]
    elif chunk_type == "content_block_delta":
        text = chunk["delta"]["text"]
    elif chunk_type == "message_delta":
        text = ""
    elif chunk_type == "message_stop":
        text = ""
    return text

def get_streaming_response(messages, streaming_callback):
    try:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": messages,
            }
        )

        # stream
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body,
        )
        stream = response.get("body")

        if stream:
            complete_response = ""
            for event in stream:  # Handle each event returned from the stream
                chunk = event.get("chunk")
                if chunk:
                    chunk_json = json.loads(chunk.get("bytes").decode())
                    if chunk_json is not None:
                        chunk_text = streaming_callback(chunk_json)
                        complete_response += chunk_text
                        yield complete_response  # Yield the accumulated response so far
            return complete_response
    except Exception as e:
        print(e, "Error occurred!")
        return ""

# Sidebar
with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With Knowledge Based RAG')
    st.button('Clear Screen', on_click=clear_screen)
    st.write(f"Total Tokens: {st.session_state.token_count}")
    
if user_prompt := st.chat_input("Ask some about some anything about ios 17"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.token_count += count_tokens(user_prompt)
    
    with st.chat_message("user"):
        st.write(user_prompt)
    
    # Prepare the conversation history for the model
    conversation_history = [
        {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
        for msg in st.session_state.messages
    ]
    
    with st.chat_message("assistant"):  
        placeholder = st.empty()
        full_response = ''
        # Get the response from the model with streaming
        for chunk in get_streaming_response(conversation_history, chunk_handler):
            full_response = chunk
            placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.token_count += count_tokens(full_response)

# Display updated token count in the sidebar
with st.sidebar:
    st.write(f"Total Tokens: {st.session_state.token_count}")

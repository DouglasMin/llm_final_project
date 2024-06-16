import boto3
from boto3.dynamodb.conditions import Key
import botocore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
import streamlit as st
import uuid
from datetime import datetime

# Setup Bedrock runtime and model
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime", 
    region_name="us-east-1",
)

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# Setup DynamoDB
TableName = "chat-messages"
dynamodb = boto3.resource("dynamodb", region_name='ap-northeast-2')
table = dynamodb.Table(TableName)

# Save conversation to DynamoDB
def save_conversation(session_id, message_id, role, content):
    try:
        timestamp = datetime.utcnow().isoformat()
        item = {
            'SessionID': session_id,
            'MessageID': str(message_id),
            'Role': role,
            'Content': content,
            'Timestamp': timestamp
        }
        table.put_item(Item=item)
        print(f"Conversation saved successfully: {item}")
    except Exception as e:
        print(f"Error saving message: {str(e)}")

# Load conversation from DynamoDB
def load_messages(session_id, last_message_id=None):
    try:
        key_condition = Key('SessionID').eq(session_id)
        
        if last_message_id is not None:
            key_condition &= Key('MessageID').gt(last_message_id)
        
        response = table.query(
            KeyConditionExpression=key_condition,
            ScanIndexForward=True  # Load messages in order of MessageID
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"Error loading messages: {str(e)}")
        return []

# Template and chain setup
template = [
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
]

prompt = ChatPromptTemplate.from_messages(template)
chain = prompt | model | StrOutputParser()

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name=TableName, session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
)

# Streamlit Setup
st.set_page_config(page_title='Streamlit Chat')

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.session_id = str(uuid.uuid4())  # Unique session ID
    st.session_state.message_id = 0  # Initialize message ID

    # Load conversation from DynamoDB
    messages = load_messages(st.session_state.session_id)
    for message in messages:
        st.session_state.messages.append({"role": message["Role"]["S"], "content": message["Content"]["S"]})

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear Chat History
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Sidebar
with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With DynamoDB Memory :brain:')
    streaming_on = st.checkbox('Streaming')
    st.button('Clear Screen', on_click=clear_screen)

# Streamlit Chat Input - User Prompt
if user_input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.message_id += 1
    with st.chat_message("user"):
        st.write(user_input)

    # Save the user message to DynamoDB
    save_conversation(st.session_state.session_id, st.session_state.message_id, "user", user_input)

    # Configure the session id
    config = {"configurable": {"session_id": st.session_state.session_id}}

    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            for chunk in chain_with_history.stream({"question": user_input}, config=config):
                full_response += chunk
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Save the assistant message to DynamoDB
            st.session_state.message_id += 1
            save_conversation(st.session_state.session_id, st.session_state.message_id, "assistant", full_response)
    else:
        # Chain - Invoke
        with st.chat_message("assistant"):
            response = chain_with_history.invoke({"question": user_input}, config=config)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Save the assistant message to DynamoDB
            st.session_state.message_id += 1
            save_conversation(st.session_state.session_id, st.session_state.message_id, "assistant", response)

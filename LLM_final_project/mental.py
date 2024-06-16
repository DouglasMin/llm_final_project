import streamlit as st
import json
import boto3
from dotenv import load_dotenv
import os

# Bedrock runtime
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

session = boto3.Session(region_name='ap-northest-2')

# Initialize a boto3 client with the provided credentials
client_polly = boto3.client('polly')

st.title("Chatbot powered by Bedrock with multi-choice model")


VOICE_OPTIONS = {
    "Salli (US)" : "Salli",
    "Kimberly (US)" : "Kimberly",
    "Mia (ESP)" : "Mia",
    
    "Seoyeon (KOR)" : "Seoyeon",
    "Matthew (US Male)": "Matthew",
    "Andrés (US ESP)" : "Andres"
}

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
        if "audio" in message:
            st.audio(message["audio"], format="audio/mp3")

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
                "system": '''
                You are a mental therapist designed to mimic the speech patterns and language of a specific person that the user wants to talk to (e.g., a friend or family member). Your goal is to create an authentic and comforting dialogue that helps the user feel supported.

                # Context
                Users seek mental therapy but prefer conversational support that feels personal and familiar. You are in the process of developing a conversational AI to replicate the chosen person’s speech patterns to enhance the user's comfort and willingness to open up.

                # Rules
                - Always respect the user's feelings and experiences.
                - Do not assume the user's emotions; respond based on their input.
                - Use appropriate language that matches the selected person’s speech style (e.g., friend, mom).
                - Ensure the conversation feels natural and supportive.
                - Maintain professionalism and empathy, even when using casual language.
                - Avoid any language or behavior that could be misinterpreted as judgmental or dismissive.

                # Instructions
                1. Identify the speech patterns and common phrases of the chosen person.
                2. Implement these patterns into the conversation dynamically.
                3. Continuously adapt the conversation based on user feedback and input.

                # Expected Input
                Users will provide details about the person they want the therapist to mimic. Anticipate a wide range of language and topics, requiring flexible and adaptive responses.

                # Output Format
                - The output should be in conversational text format.
                - Responses should be concise but meaningful, typically "3 to 4 sentences" per response.
                - The conversation that you have to have with the user should be seamless.
                - Unless the user says quotes intended to end the conversation, you should not end the conversation"
                - For terms such as to describe the verbal response,
                e.g. 'in soothing voice, mom' put them between bracket.
                ''',
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
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

def synthesize_speech(text, filename, voice):
    try:
        # Passing the text to Amazon Polly
        response_voice = client_polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId=voice  # You can change the voice here
        )

        # Saving the audio
        if "AudioStream" in response_voice:
            with open(filename, "wb") as file:
                file.write(response_voice['AudioStream'].read())
            print(f"Audio file saved as {filename}")
            return filename
        else:
            print("Could not stream audio")
            return None
    except Exception as e:
        print(e, "Error occurred during speech synthesis!")
        return None

# Sidebar
with st.sidebar:
    st.title('Streamlit Chat')
    st.subheader('With Knowledge Based RAG')
    st.button('Clear Screen', on_click=clear_screen)
    st.write(f"Total Tokens: {st.session_state.token_count}")
    selected_voice = st.selectbox("Choose a voice", list(VOICE_OPTIONS.keys()))


if user_prompt := st.chat_input("Ask something about iOS 17"):
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
        audio_file = ''
        # Get the response from the model with streaming
        for chunk in get_streaming_response(conversation_history, chunk_handler):
            full_response = chunk
            placeholder.markdown(full_response)
        
        # Save the full response and generate audio
        audio_filename = f"output_{len(st.session_state.messages)}.mp3"
        if full_response:
            audio_file = synthesize_speech(full_response, audio_filename, VOICE_OPTIONS[selected_voice])

        # Append the response and audio file to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response, "audio": audio_file})
        st.session_state.token_count += count_tokens(full_response)
        
        # Display audio player if audio file exists
        if audio_file:
            st.audio(audio_file, format="audio/mp3")

# Display updated token count in the sidebar
with st.sidebar:
    st.write(f"Total Tokens: {st.session_state.token_count}")

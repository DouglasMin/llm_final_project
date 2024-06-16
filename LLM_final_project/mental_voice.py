import boto3
import os
from dotenv import load_dotenv

load_dotenv()

session = boto3.Session(region_name='ap-northest-2')

# Initialize a boto3 client with the provided credentials
client_polly = boto3.client('polly')


# Text to synthesize
text = """
 Hello pleasure to meet you.

"""


def voice():
    response = client_polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli'  # You can change the voice here
    )
    
    # Saving the audio
    if "AudioStream" in response:
        with open("output.mp3", "wb") as file:
            file.write(response['AudioStream'].read())
        print("Audio file saved as output.mp3")
    else:
        print("Could not stream audio")
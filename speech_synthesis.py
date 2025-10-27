import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
import numpy as np
from io import BytesIO

# Page setup
st.set_page_config(page_title="Text to Speech Generator", page_icon="🗣️")
st.title("🎙️ Text to Speech using Microsoft SpeechT5")

@st.cache_resource
def load_models():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder

processor, model, vocoder = load_models()

# Random speaker embedding
speaker_embedding = torch.randn((1, 512))

# User input
text = st.text_area("Enter text to convert to speech:", "Hello boss, this is Jarvis speaking!")

if st.button("Generate Speech 🎧"):
    with st.spinner("Generating speech..."):
        inputs = processor(text=text, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

        # Increase volume
        speech_np = speech.numpy() * 2.0  # 🔊 2x louder (safe amplification)
        speech_np = np.clip(speech_np, -1.0, 1.0)

        # Save to buffer
        buffer = BytesIO()
        sf.write(buffer, speech_np, 16000, format="WAV")
        buffer.seek(0)

        # Play and download
        st.audio(buffer, format="audio/wav")
        st.download_button("💾 Download Audio", buffer, file_name="tts_output.wav", mime="audio/wav")

        st.success("✅ Speech generated successfully!")

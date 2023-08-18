import streamlit as st
import huggingface_hub 
from llama_cpp import Llama

st.title("Llama 2 Demo")

MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-GGML" 
MODEL_BASENAME = "llama-2-13b-chat.ggmlv3.q5_1.bin"

@st.cache_data
def load_model():
   model_path = huggingface_hub.hf_hub_download(repo_id=MODEL_NAME_OR_PATH, filename=MODEL_BASENAME)
   return Llama(model_path=model_path) 

llama = load_model()

prompt = st.text_input("Enter a prompt:", "What is machine learning?")

if st.button("Generate"):
   prompt_template = f"""
   SYSTEM: You are helpful, respectful, and honest.
   
   USER: {prompt} 
   
   ASSISTANT:
   """

   response = llama(prompt=prompt_template, max_tokens=100, temperature=0.5)
   st.write(response["choices"][0]["text"])
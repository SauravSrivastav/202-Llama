import streamlit as st
import langchain
import transformers
from transformers import AutoTokenizer
from huggingface_hub import login

# Log in to HuggingFace  
login("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")

# Load model and tokenizer
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)

# Create pipeline and langchain objects 
pipeline = transformers.pipeline("text-generation", 
                model=model,
                tokenizer=tokenizer)

llm = langchain.HuggingFacePipeline(pipeline)

template = """
             Create a SQL query snippet using the below text:
              ```{text}```
              Just SQL query:
           """

prompt = langchain.PromptTemplate(template=template, input_variables=["text"])

llm_chain = langchain.LLMChain(prompt=prompt, llm=llm)

st.title("SQL Query Generator")

text = st.text_input("Enter text:", value="")
if text:
    response = llm_chain.run(text)
    st.code(response)
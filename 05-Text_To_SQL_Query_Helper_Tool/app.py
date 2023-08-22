import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate,  LLMChain

# Load the local Llama model using the CTransformers class
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})

# Create a text input for the user to enter the text to generate SQL query for

template = """
             Create a SQL query snippet using the below text:
              ```{text}```
              Just SQL query:
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

text = st.text_input("Enter text to generate SQL query for:")

# Generate SQL query using the Llama model when the user clicks the "Generate" button
if st.button("Generate"):
    result = llm.generate([f"Create a SQL query snippet using the below text: ```{text}``` Just SQL query:"])
    st.write(result)

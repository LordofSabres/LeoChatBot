import streamlit as st
import torch
import json
from utils.constants import *

# LlamaIndex imports
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding

# LangChain embeddings (fixed)
from langchain.embeddings import HuggingFaceEmbeddings

# IBM Watson
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

st.title("💬 Chat with My AI Assistant")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style/styles_chat.css")

# Load variables from constants
pronoun = info['Pronoun']
name = info['Name']
subject = info['Subject']
full_name = info['Full_Name']

# Chat history
if "messages" not in st.session_state:
    welcome_msg = f"Hi! I'm {name}'s AI Assistant, Leo. How may I assist you today?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# Sidebar UI
with st.sidebar:
    st.markdown("# Chat with my AI assistant")
    with st.expander("Click here to see FAQs"):
        st.info(f"""
        - What are {pronoun} strengths and weaknesses?
        - What is {pronoun} expected salary?
        - What is {pronoun} latest project?
        - When can {subject} start to work?
        - Tell me about {pronoun} professional background
        - What is {pronoun} skillset?
        - What is {pronoun} contact?
        - What are {pronoun} achievements?
        """)
    st.download_button(
        label="Download Chat",
        data=json.dumps(st.session_state.messages),
        file_name='chat.json',
        mime='application/json'
    )
    st.caption(f"© Made by {full_name} 2025. All rights reserved.")

# LLM + Embedding Initialization
with st.spinner("Initiating the AI assistant. Please hold..."):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def init_llm():
        credentials = {
            'url': "https://us-south.ml.cloud.ibm.com",
            'apikey': "eOjQXQsmyOnr_YFjKUEyGyxQG7GTCNQxsEL6rwr99g5P"
        }

        params = {
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1
        }

        model = Model(
            model_id="ibm/granite-13b-instruct-v2",
            credentials=credentials,
            params=params,
            project_id="2ee58a27-be42-4e80-87e7-c2f885da2287"
        )

        watsonx_llm = WatsonxLLM(model=model)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": DEVICE}
        )

        return watsonx_llm, embeddings

    watsonx_llm, embeddings = init_llm()

    documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()
    llm_predictor = LLMPredictor(llm=watsonx_llm)
    embed_model = LangchainEmbedding(embeddings)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

def ask_bot(user_query):
    PROMPT_QUESTION = f"""You are Leo, an AI assistant dedicated to assisting {name} in {pronoun} job search by providing recruiters with relevant information about {pronoun} qualifications and achievements. 
Your goal is to support {name} in presenting {pronoun}self effectively to potential employers and promoting {pronoun} candidacy for job opportunities.
If you do not know the answer, politely admit it and let recruiters know how to contact {name} directly.
Don't say 'Leo:' or add a breakline at the start.
Human: {user_query}"""
    return index.as_query_engine().query(PROMPT_QUESTION)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = ask_bot(prompt)
            st.write(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

questions = [
    f'What are {pronoun} strengths and weaknesses?',
    f'What is {pronoun} latest project?',
    f'When can {subject} start to work?'
]

def send_button_ques(question):
    st.session_state.disabled = True
    response = ask_bot(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response.response})

if 'button_question' not in st.session_state:
    st.session_state['button_question'] = ""
if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False

if not st.session_state['disabled']:
    for n, msg in enumerate(st.session_state.messages):
        if n == 0:
            with st.container():
                for q in questions:
                    st.button(label=q, on_click=send_button_ques, args=[q], disabled=st.session_state.disabled)

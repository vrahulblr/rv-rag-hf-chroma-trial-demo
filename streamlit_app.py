import streamlit as st

import json
import requests
import time

from newspaper import Article

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA

from langchain.document_loaders import UnstructuredHTMLLoader

from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter

st.title("Simple app to demonstrate Retrieval Augmented Generation, fondly known as RAG")
st.markdown("Enter a URL that you want the app to use as context, and the question you want answered")

# Here we are collecting user inputs
with st.form("user_inputs_for_rag_form", clear_on_submit=False):
    API_KEY = st.text_input("Insert your Hugging Face Access Token here")
    url = st.text_input("Insert the URL you would like to provide as context")
    question = st.text_input("Your question here")
    submit = st.form_submit_button("Submit")

title = []
text = []
answer_to_print = []

#We will use this function later to call the LLM to generate content
def query(API_URL, headers, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# We scrape the news site to collect the content over there
if submit:
    article = Article(url)
    article.download()
    article.parse()
    title.append(article.title)
    text.append(article.text)

try:
    doc_creator = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    document = doc_creator.create_documents(texts = text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(document)

    # Indicate which model you wish to use for creating embeddings
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    # Arguements to send alongside the model configs
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs 
    )
    db = FAISS.from_documents(docs, embeddings)
    searchDocs = db.similarity_search(question)
    answer_to_print.append(searchDocs[0].page_content)
    st.divider()
    st.write(f"The relevant document (chunk) from filing numbered **{title[0]}**")
    st.write(f"**{answer_to_print[0]}**")
except Exception as e:
    pass


try:
    #We will use the Phi3 model to generate an answer to our prompt
    API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    #Here we are creating the prompt using a template corresponding to the model we will use
    message = [
        {"role": "system", "content": "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users. Respond back in 1 sentence. Here is the context: " + f"{answer_to_print[0]}" + "Now answer the following question from the user"},
        {"role": "user", "content": f"{question}"},
    ]
    headers = {"Authorization": f"Bearer {API_KEY}"}
    output = query(API_URL, headers, {"inputs": str(message),"wait_for_model": True})
    st.divider()
    st.subheader("The LLM has the following answer to your question, based on context retrieved from your URL")
    st.write(output[0]['generated_text'])
    st.divider()
    st.markdown("Remember to include an Output Parser to get a neatly formatted answer")
except Exception as e:
    pass
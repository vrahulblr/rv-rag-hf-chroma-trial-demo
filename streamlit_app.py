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

# question = st.text_input("Enter a question that you want answered from the filing", type="default")

st.title("Query an SEC filing by a listed company")
st.markdown("**Insert HTML links to SEC filings (like 10Ks, 10Qs from companies) and ask questions.**")
st.subheader("Enter the URL of filing you wish to query")
user_input = st.text_input("Enter the URL and then the question, separated by comma", type="default")
fetch_button = st.button("Fetch answer")
try:
    user_input_as_list = user_input.split(",")
    API_KEY = user_input_as_list[0]
    url = user_input_as_list[1]
    question = user_input_as_list[2]
except Exception as e:
    print("Awaiting user input")


title = []
text = []
# db = []


def query(API_URL, headers, payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()



answer_to_print = []

if fetch_button:
    article = Article(url)
    article.download()
    article.parse()
    title.append(article.title)
    text.append(article.text)
    print("type and length of text", type(text[0]), len(text[0]))

try:
    doc_creator = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    document = doc_creator.create_documents(texts = text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(document)
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    print("embeddings done, proceeding to write in Vector DB")
    db = FAISS.from_documents(docs, embeddings)
    print("written to vectordb")
    searchDocs = db.similarity_search(question)
    answer_to_print.append(searchDocs[0].page_content)
    st.divider()
    st.write(f"The relevant document (chunk) from filing numbered **{title[0]}**")
    st.write(f"**{answer_to_print[0]}**")
except Exception as e:
    pass


try:
    # print(type(answer_to_print[0]), len(answer_to_print[0]), answer_to_print)
    message = [
        {"role": "system", "content": "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users. Respond back in 1 sentence. Here is the context: " + f"{answer_to_print[0]}" + "Now answer the following question from the user"},
        {"role": "user", "content": f"{user_input_as_list[2]}"},
    ]
    API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    output = query(API_URL, headers, {"inputs": str(message),"wait_for_model": True})
    st.divider()
    st.write(output[0]['generated_text'].split("]")[1])
except Exception as e:
    pass
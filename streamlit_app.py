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
    url = user_input_as_list[0]
    question = user_input_as_list[1]
except Exception as e:
    print("Awaiting user input")
# print("URL is", url)
# print("question is", question)
# title = 'sample title'
# text = 'sample text'

title = []
text = []
# db = []

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
    answer_to_print = searchDocs[0].page_content
    st.divider()
    st.write(f"Your answer from filing numbered **{title[0]}**")
    st.write(f"**{answer_to_print}**")
except Exception as e:
    pass
# print("type of db, and of first entity in db", type(db), type(db[0]), "length", len(db))

# question = "How many retail stores Nike has?"


# if submit_button:
#     print("submit button hit recorded", question)
#     try:
#         dbref = db[0]
#         searchDocs = dbref.similarity_search(question)
#         print("semantic search underway")
#         answer_to_print = searchDocs[0].page_content
#         st.divider()
#         st.write(f"Your answer from filing numbered **{title[0]}**")
#         st.write(f"**{answer_to_print}**")
#     except Exception as e:
#         print("exception on submit button")
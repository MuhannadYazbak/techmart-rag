# app/interface.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
import os
from dynamic_agent import build_agent

agent = build_agent(2)
response = agent.invoke("Hello, did I ordered any smart watch ?")
print(response)
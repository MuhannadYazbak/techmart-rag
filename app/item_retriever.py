from sqlalchemy import create_engine
import pandas as pd
from langchain_community.vectorstores import FAISS
# 💡 Swap out the local heavy HuggingFace engine for a lightweight API connector
from langchain_openai import OpenAIEmbeddings 
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

def get_item_retriever():
    db_url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}??ssl_disabled=false"

    engine = create_engine(db_url, 
        connect_args={"ssl": {"check_hostname": False}})
    query = "SELECT name, price, description, category, quantity FROM itemtable"
    df = pd.read_sql(query, engine)
    
    docs = [
        Document(
            page_content=f"Name: {row['name']}\nPrice: {row['price']}\nDescription: {row['description']}\nCategory:{row['category']}\nQuantity: {row['quantity']}",
            metadata={"name": row["name"]}
        )
        for _, row in df.iterrows()
    ]

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 💡 Use cloud API embeddings. (Uses zero container memory or disk space!)
    # This automatically reads your OPENAI_API_KEY from your environment.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()
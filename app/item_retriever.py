import os  
import urllib.parse  # 💡 Escapes special symbols in passwords safely
from sqlalchemy import create_engine
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def get_item_retriever():
    user = os.getenv('DB_USER')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT', '3306')  
    name = os.getenv('DB_NAME')
    
    # 💡 Explicitly capture and encode the password variable cleanly
    raw_password = os.getenv('DB_PASS', '')
    password = urllib.parse.quote_plus(raw_password)

    # 💡 Now 'password' matches exactly inside the connection URL layout
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
    
    engine = create_engine(
        db_url,
        connect_args={"ssl": {"check_hostname": False}}
    )
    
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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()
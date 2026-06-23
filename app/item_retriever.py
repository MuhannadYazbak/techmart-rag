import os 
from sqlalchemy import create_engine
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings 
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def get_item_retriever():
    # 💡 Pull all components dynamically from Render Environment Variables
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT', '3306')
    name = os.getenv('DB_NAME')

    # 💡 Keep the URL completely clean without string-level parameters
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
    
    # 💡 Pass the secure configuration explicitly here so PyMySQL handles it natively
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
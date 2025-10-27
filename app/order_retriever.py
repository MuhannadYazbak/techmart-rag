from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

def get_orders_df(user_id):
    db_url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    query = f"SELECT * FROM ordertable WHERE user_id = {user_id}"
    return pd.read_sql(query, engine)



def orders_to_documents(df):
    docs = []
    for _, row in df.iterrows():
        items = json.loads(row["items_json"])
        item_lines = "\n".join([f"- {item['name']} x{item['quantity']} ({item['price']} ILS)" for item in items])
        content = (
            f"Order ID: {row['order_id']}\n"
            f"Status: {row['status']}\n"
            f"Total: {row['total_amount']} ILS\n"
            f"Created At: {row['created_at']}\n"
            f"Items:\n{item_lines}"
        )
        docs.append(Document(page_content=content, metadata={"order_id": row["order_id"]}))
    return docs



def get_orders_retriever(user_id):
    df = get_orders_df(user_id)
    docs = orders_to_documents(df)

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()
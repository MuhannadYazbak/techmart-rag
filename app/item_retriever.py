from sqlalchemy import create_engine
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

def get_item_retriever():
    
    db_url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"

    # Create SQLAlchemy engine
    engine = create_engine(db_url)
    # Load product data
    query = "SELECT name, price, description, quantity FROM itemtable"
    df = pd.read_sql(query, engine)
    

    # Convert rows to LangChain Documents
    docs = [
        Document(
            page_content=f"Name: {row['name']}\nPrice: {row['price']}\nDescription: {row['description']}\nQuantity: {row['quantity']}",
            metadata={"name": row["name"]}
        )
        for _, row in df.iterrows()
    ]

    # Split and embed
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()
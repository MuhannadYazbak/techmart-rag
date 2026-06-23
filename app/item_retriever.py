def get_item_retriever():
    # 💡 Pull all necessary components from your Render Environment Variables
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT', '3306') # Safely grabs 15411 from Render
    name = os.getenv('DB_NAME')

    # 💡 Structure the complete URL with port and SSL enforcement
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?ssl_mode=REQUIRED"

    engine = create_engine(db_url)
    
    # --- The rest of your code stays exactly identical ---
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
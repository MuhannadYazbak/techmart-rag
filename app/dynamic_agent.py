from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from app.item_retriever import get_item_retriever
from app.order_retriever import get_orders_retriever
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

# Define the prompt
prompt = PromptTemplate.from_template("""
You are TechMart's friendly assistant. Use the following context to answer customer questions about products or orders.

Context:
{context}

Question:
{question}

Answer as a helpful, conversational assistant.
""")

# Initialize the LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://techmart.ai",
        "X-Title": "TechMart Assistant"
    }
)

def classify_query(query):
    if any(word in query.lower() for word in ["order", "shipment", "delivery", "tracking", "status", "طلبية", "شحنة", "توصيل", "הזמנה", "משלוח", "סטטוס"]):
        return "order"
    return "product"

def build_agent(user_id=None):
    item_retriever = get_item_retriever()
    order_retriever = get_orders_retriever(user_id)

    def dynamic_retriever(query):
        if classify_query(query) == "order":
            return order_retriever.invoke(query)
        return item_retriever.invoke(query)

    rag_chain = (
        RunnableMap({"context": RunnableLambda(dynamic_retriever), "question": RunnablePassthrough()})
        | prompt
        | llm
    )

    return rag_chain
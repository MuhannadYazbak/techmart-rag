from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from item_retriever import get_item_retriever
from order_retriever import get_orders_retriever

import os
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

prompt = PromptTemplate.from_template("""
You are TechMart's friendly assistant seller. Use the following context details about our product inventory and the customer's order history to answer their questions accurately.

Context:
{context}

Question:
{question}

Answer as a helpful, conversational assistant.
""")

def build_agent(user_id: int):
    # 💡 Load both dynamic contextual layers
    item_retriever = get_item_retriever()
    order_retriever = get_orders_retriever(user_id)

    # Combine retrievers so the LLM gets a complete view of items + orders
    def combined_context(query_str):
        item_docs = item_retriever.invoke(query_str)
        order_docs = order_retriever.invoke(query_str)
        
        context_text = "--- PRODUCT INVENTORY ---\n"
        context_text += "\n\n".join([d.page_content for d in item_docs])
        context_text += "\n\n--- CUSTOMER ORDER HISTORY ---\n"
        context_text += "\n\n".join([d.page_content for d in order_docs])
        return context_text

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        default_headers={
            "HTTP-Referer": "https://techmart.ai",
            "X-Title": "TechMart Assistant"
        }
    )

    rag_chain = (
        RunnableMap({
            "context": lambda x: combined_context(x["question"]), 
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
    )

    return rag_chain
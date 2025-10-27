from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
from retriever import get_retriever
from item_retriever import get_item_retriever
from order_retriever import get_orders_retriever

import os
from dotenv import load_dotenv

load_dotenv()

# Load OpenRouter API key
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Define prompt
prompt = PromptTemplate.from_template("""
You are TechMart's friendly assistant seller. Use the following context to answer customer questions about products, pricing, and availability.

Context:
{context}

Question:
{question}

Answer as a helpful, conversational assistant.
""")

# Build the agent
def build_agent():
    retriever = get_item_retriever()

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        default_headers={
            "HTTP-Referer": "https://techmart.ai",  # Your site or project name
            "X-Title": "TechMart Assistant"
        }
    )

    rag_chain = (
        RunnableMap({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
    )

    return rag_chain
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_groq import ChatGroq

load_dotenv()

def baseline_query(pdf_path: str, web_url: str, question: str):
    # Load a PDF (e.g. terms of service)
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_pages = pdf_loader.load()
    pdf_text = "\n".join(p.page_content for p in pdf_pages)

    # Load a webpage (e.g. pricing page)
    web_loader = WebBaseLoader(web_url)
    web_docs = web_loader.load()
    web_text = "\n".join(d.page_content for d in web_docs)

    # Stuff BOTH sources into the prompt, no chunking, no retrieval yet
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt = f"""You are answering questions using company documentation.
Based ONLY on the text below, answer the question. If the answer isn't
in the text, say so clearly.

--- PDF DOCUMENT ---
{pdf_text}

--- WEBPAGE CONTENT ---
{web_text}

QUESTION: {question}
"""
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    answer = baseline_query(
        pdf_path="data/netflix.pdf",
        web_url="https://www.netflix.com/signup/planform",
        question="What are Netflix's subscription plans and prices, and what does the terms of use say about cancellation?"
    )
    print(answer)
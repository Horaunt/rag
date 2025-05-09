import re
import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate

# Load vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# LLM
llm = Ollama(model="mistral")

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Use the context below to answer the user's question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

# Dictionary API
def define_word(term):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
    res = requests.get(url)
    if res.status_code == 200:
        meaning = res.json()[0]["meanings"][0]["definitions"][0]["definition"]
        return f"Definition of '{term}': {meaning}"
    return f"Could not find a definition for '{term}'."

# Calculator
def evaluate_expression(expr):
    try:
        safe_expr = re.sub(r"[^0-9+-\*/(). ]", "", expr)
        return f"Result: {eval(safe_expr)}"
    except Exception:
        return "Invalid mathematical expression."

# Streamlit UI
st.title("RAG-Powered Q&A Assistant")
st.markdown("""
This is a simple Q&A assistant powered by RAG (Retrieve and Generate) combined with a calculator and dictionary.
Ask a question, and the assistant will provide an answer by retrieving context or using the calculator/dictionary.
""")

query = st.text_input("Ask a question:")

if query:
    # 🔀 Agentic Routing
    if "calculate" in query.lower() or re.search(r"\d+ *[\+\-\*/] *\d+", query):
        route = "calculator"
        st.subheader("🧮 Routed to: Calculator")
        st.write(evaluate_expression(query))

    elif query.lower().startswith("define ") or query.lower().startswith("what is "):
        route = "dictionary"
        st.subheader("📚 Routed to: Dictionary")
        term = query.split(" ", 1)[-1]
        st.write(define_word(term))

    else:
        route = "rag"
        st.subheader("📄 Routed to: RAG + Ollama")
        docs = vectordb.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Generate prompt
        prompt = prompt_template.format(context=context, question=query)

        # LLM response
        response = llm.invoke(prompt)

        # Display retrieved context and final answer
        st.write("🔍 **Top Retrieved Chunks:**")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content.strip()[:400]}")

        st.write("\n💬 **Final Answer:**")
        st.write(response.strip())


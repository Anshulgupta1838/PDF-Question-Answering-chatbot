from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st 
from time import sleep

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def Document_process(path):
  ##Document loader
    loader = PyPDFLoader(path)
    docs = loader.load()
  ## Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = splitter.split_documents(docs)
  #Embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)
    
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


st.subheader("📄Document Q&A ChatBot - Aak Anything")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

## Document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader("📄Select your PDF File", type="pdf")

    if file and st.button("🚀 Process Document"):
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing PDF..."):
            Document_process("./uploaded_document.pdf")

        st.markdown("✅Document Uploaded Successfully...")
        sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anythings...")
    if query:

        st.session_state.messages.append({"role":"user", "content":"query"})
        
        st.chat_message("user").markdown(query)

        documents = st.session_state.vector_db.similarity_search(query, k=2 )
        context = ""

        for doc in documents:
            context += doc.page_content + "\n\n"

        
        # Create prompt
        prompt = f"""
        You are a helpful PDF Question Answering Assistant.

        Answer the user's question using ONLY the provided PDF context.

        Instructions:
        1. Give direct and accurate answers.
        2. Keep answers short and easy to understand.
        3. If the question asks for a name, give only the name.
        4. Avoid unnecessary explanations.
        5. Do not include technical metadata or signatures.
        6. If the answer is not available in the PDF,
        say "Information not found in the PDF."

        PDF Context:
        {context}

        User Question:
        {query}

        Answer:
        """

        # Generate response
        result = llm.invoke(prompt)

        # Extract only text from Gemini response
        if isinstance(result.content, str):
            answer = result.content

        elif isinstance(result.content, list):
            answer_parts = []

            for item in result.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        answer_parts.append(item.get("text", ""))

                elif isinstance(item, str):
                    answer_parts.append(item)

            answer = "\n".join(answer_parts).strip()

        else:
            answer = str(result.content)

        # Save clean answer in chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # Display only clean answer
        with st.chat_message("assistant"):
            st.markdown(answer)


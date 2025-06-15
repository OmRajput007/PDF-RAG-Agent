import streamlit as st
import os
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonAstREPLTool
from langchain.tools import Tool
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

# PDF paths
pdf_paths = ["converted_text.pdf", "robert_greene.pdf"]
all_docs = []

# Title
st.title("PDF AI Agent")

# Intialised
if 'intialised' not in st.session_state :

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunks = splitter.split_documents(documents)
        all_docs.extend(chunks)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local("pdf_vectorstore")

    def query_pdf_tool(input_text = str) -> str :
        db = FAISS.load_local("pdf_vectorstore", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(input_text, k=3)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        return chain.run(input_documents = docs, question = input_text)

    pdf_tool = Tool(
        name = "PDF assistant",
        func = query_pdf_tool,
        description="Answers questions from PDFs"
    ) 


    # AGENT COMES INTO PICTURE
    llm = OpenAI(temperature=0)
    tools = [pdf_tool, PythonAstREPLTool()]
    prompt = hub.pull ("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=True)

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages :
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    if prompt := st.chat_input("Ask something about your PDFs : "):
        st.session_state.messages.append({"role":"user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent_executor.invoke({"input": prompt})
                answer = response["output"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        


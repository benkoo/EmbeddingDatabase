import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import ollama
import os
from dotenv import load_dotenv

load_dotenv() 

# Retrieve the system-wide LLM model name
model_name = os.getenv("MODEL_NAME_FOR_EMBEDDINGS")
embeddings = OllamaEmbeddings(model=model_name)


# SOME DEFAULT SETTINGS
language_choice = '中文'
assistant_text = "盡可能不要回答超過Context中提供的信息範圍，如果不知道答案，就說不知道"

def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {'使用' + language_choice +'回答。' + context }"
    response = ollama.chat(
                    model=model_name,
                    messages=[
                            {'role': 'system', 'content': formatted_prompt},
                            {'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': assistant_text}],
                    )
    return response['message']['content']

from datetime import datetime
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H")

persist_directory = 'EmbeddingDBs/' + time_str + model_name

# Create the vector store
vectordb = Chroma(
    embedding_function = embeddings,
    persist_directory=persist_directory
)


retriever = vectordb.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)



def retrieverQA(input):
    return "Dummy Text front " + input + " Dummy Text back"

demo = gr.Interface(
        fn = rag_chain,
        inputs='text',
        outputs='text',
        title = "Loading Data from " + persist_directory
)

demo.launch(server_name='0.0.0.0')
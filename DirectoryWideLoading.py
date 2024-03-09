from langchain_community.document_loaders import PyPDFLoader
import os 
from dotenv import load_dotenv
load_dotenv() 

# Define a directory name
model_name = os.getenv("MODEL_NAME_FOR_EMBEDDINGS")
directory_name = 'data' 
pdf_files = []

pdf_files = [f for f in os.listdir(directory_name) if f.endswith('.pdf')]

print(pdf_files)        

docs = []

for f in pdf_files:
    pages = PyPDFLoader(directory_name + '/' + f).load()
    docs.extend(pages)
    
    
print("Total Number of Documents:" + str(len(docs)))    

print(docs[1])    

from langchain.text_splitter import CharacterTextSplitter


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=100,
    length_function=len
)

docs = text_splitter.split_documents(docs)

from langchain_community.embeddings import OllamaEmbeddings


embeddings = OllamaEmbeddings(model=model_name)

from datetime import datetime
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H")

persist_directory = 'EmbeddingDBs/' + time_str + model_name

from langchain_community.vectorstores import Chroma

# Create the vector store
vectordb = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory=persist_directory
)
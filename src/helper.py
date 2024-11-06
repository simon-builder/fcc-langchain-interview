from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
# from .prompt import *

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def file_processing(file_path):
    # load the file
    loader = PyPDFLoader(file_path)
    doc = loader.load()
    
    # put all pages into a single string
    pages_in_one_string = ''
    
    for page in doc:
        pages_in_one_string += page.page_content
    
    # split the string into chunks
    text_splitter = TokenTextSplitter(model_name="gpt-4o-mini", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(pages_in_one_string)
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
from prompt import prompt_template, refine_template

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
    text_splitter = TokenTextSplitter(model_name="gpt-4o-mini", chunk_size=8000, chunk_overlap=200)
    string_in_chunks = text_splitter.split_text(pages_in_one_string)
    
    # converting into documents for embedding
    string_in_docs = [Document(page_content=chunk) for chunk in string_in_chunks]
    
    # split the documents into smaller chunks to overcome the token limit from model
    doc_splitter = TokenTextSplitter(model_name="gpt-4o-mini", chunk_size=1000, chunk_overlap=200)
    docs_in_chunks = doc_splitter.split_documents(string_in_docs)
    
    return string_in_docs, docs_in_chunks

def llm_pipeline(file_path):
    # Step 1: Process the PDF file into two different chunk sizes
    large_chunks, small_chunks = file_processing(file_path)  # Rename for clarity
    
    # Question Generation Pipeline - uses larger chunks for better context
    chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    prompt_questions = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_prompt_questions = PromptTemplate(template=refine_template, input_variables=["existing_answer", "text"])
    
    question_chain = load_summarize_chain(
        llm=chat_model,
        chain_type="refine",
        verbose=True,
        question_prompt=prompt_questions,
        refine_prompt=refine_prompt_questions
    )
    generated_questions = question_chain.invoke(large_chunks)
    
    # Answer Generation Pipeline - uses smaller chunks for precise retrieval
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(small_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return generated_questions, qa_chain

generated_questions, answer_generation_chain =llm_pipeline('data/SDG.pdf')

## Continue at 11:20:23 creating endpoint
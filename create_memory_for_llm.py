from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Load a new PDF
DATA_PATH = "data/"
def load_pdf_documents(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_documents(data=DATA_PATH)
#print("Length of loaded documents: ", len(documents))

#Split the documents into smaller chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

chunks = create_chunks(extracted_data=documents)

#print("Length of loaded chunks: ", len(chunks))

#Create vector embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = get_embedding_model()

#Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
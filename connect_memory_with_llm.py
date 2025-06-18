import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
#Setup LLM Mistral
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm

#Connect llm with FAISS and create chain
CUSTOM_PROMPT_TEMPLATE="""
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer then try guessing the nearest possible answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Do a little small talk and then answer
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

#load database
DB_FAISS_PATH = "vectorstore/db_faiss" 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

#Invoke the chain with a query
user_query = input("Write a question here: ")
response = qa_chain.invoke({'query': user_query})

print("Result: ", response['result'])
print("Source Documentes: ", response['source_documents'])
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()


# 1. Data Extraction
def load_web_data():
    url = "https://brainlox.com/courses/category/technical"
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    return docs

# 2. Create Embeddings and Vector Store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Initialize Grok LLM
def initialize_llm():
    os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY")
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",  
        temperature=0.7
    )

app = Flask(__name__)

docs = load_web_data()
vector_store = create_vector_store(docs)
llm = initialize_llm()


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# 3. Flask RESTful API Endpoints
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        result = qa_chain({"query": query})
        response = {
            'answer': result['result'],
            # 'sources': [
            #     {
            #         'content': doc.page_content[:200],  
            #         'metadata': doc.metadata
            #     }
            #     for doc in result['source_documents']
            # ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
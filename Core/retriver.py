from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_visie_retriever(top_k: int = 5):
    """
    Creates a retriever from the markdown-based chroma_visie_tech collection.

    Args:
        top_k (int): Number of documents to retrieve. Defaults to 5.
        
    Returns:
        A retriever object that can perform similarity search on the visie_tech collection.
    """
    # Get Google API key from environment
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Initialize embeddings model with correct model name
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    # Define the database path for markdown-based embeddings
    db_path = "DataLoader/chroma_db/chroma_visie_tech"
    
    # Check if database path exists
    if not os.path.exists(db_path):
        # List directory contents for debugging
        if os.path.exists("DataLoader/"):
            print("Contents of DataLoader/:")
            for item in os.listdir("DataLoader/"):
                print(f"  {item}")
                if os.path.isdir(f"DataLoader/{item}") and item == "chroma_db":
                    chroma_items = os.listdir(f"DataLoader/{item}")
                    for chroma_item in chroma_items:
                        print(f"    {chroma_item}")
        
        raise FileNotFoundError(f"Database path not found: {db_path}")
    
    # Create the collection name for markdown embeddings
    collection_name = "markdown_001_visie_tech"
    
    vectorstore = Chroma(
        collection_name=collection_name, 
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    # Get document count
    try:
        count = vectorstore._collection.count()
        print(f"Successfully connected to markdown collection '{collection_name}' with {count} documents")
    except Exception as e:
        print(f"Warning: Could not get document count: {e}")
    
    # Create retriever with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    return retriever

def search_visie_info(query: str, top_k: int = 5) -> List[Document]:
    """
    Searches for VisieInfo in the markdown-based collection.

    Args:
        query (str): The search query
        top_k (int): Number of results to return
        
    Returns:
        List[Document]: List of relevant documents
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    try:
        # Get the retriever for markdown-based embeddings
        retriever = get_visie_retriever(top_k)
        results = retriever.get_relevant_documents(query)
        
        # Add source information to metadata
        for doc in results:
            if hasattr(doc, 'metadata'):
                doc.metadata['source_collection'] = 'visie_tech_markdown'
                doc.metadata['embedding_type'] = 'markdown'
            else:
                doc.metadata = {
                    'source_collection': 'visie_tech_markdown',
                    'embedding_type': 'markdown'
                }
        
        print(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
        return results
        
    except Exception as e:
        print(f"Error searching visie info: {e}")
        return []

def get_employees_retriever(top_k: int = 3):
    """
    Creates a retriever for the employees information collection.

    Args:
        top_k (int): Number of results to return
        
    Returns:
        A retriever object that can perform similarity search on the employees information collection.
    """
    # Get Google API key from environment
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Initialize embeddings model with API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key
    )

    # Connect to the existing Chroma collection
    db_path = "DataLoader/Chroma_db/VisieEmployee/"
    
    # Check if database path exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path does not exist: {db_path}")
    
    vectorstore = Chroma(
        collection_name="employees_info",
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    # Create retriever with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    return retriever

def search_employees_info(query: str, top_k: int = 3) -> List[Document]:
    """
    Searches for employee information in the employees information collection.

    Args:
        query (str): The search query to find relevant employee information
        top_k (int): Number of results to return
        
    Returns:
        List[Document]: List of relevant documents with employee information
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Get the employees retriever
    retriever = get_employees_retriever(top_k)

    # Get relevant documents
    results = retriever.get_relevant_documents(query)

    # Add source information to metadata
    for doc in results:
        if hasattr(doc, 'metadata'):
            doc.metadata['source_collection'] = 'employees_info'
        else:
            doc.metadata = {'source_collection': 'employees_info'}

    return results




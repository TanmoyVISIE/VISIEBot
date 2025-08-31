from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os

def get_visie_retriever(collection_name: str, top_k: int = 3):
    """
    Creates a retriever for the specified Chroma collection of VisieInfo.

    Args:
        collection_name (str): Name of the collection to retrieve from.
                              Should be one of: "About", "AI Insights", "Contact", "Documind", "Kothok", "Papers", "Percept", "Verifyid" or "AI solutions for Innovation".
        top_k (int): Number of documents to retrieve. Defaults to 3.
        
    Returns:
        A retriever object that can perform similarity search on the specified collection.
    """
    # Validate collection name
    valid_collections = ["About", "AI Insights", "Contact", "Documind", "Kothok", "Papers", "Percept", "Verifyid", "AI solutions for Innovation"]
    if collection_name not in valid_collections:
        raise ValueError(f"Invalid collection name: {collection_name}. Must be one of: {valid_collections}")
    
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Connect to the existing Chroma collection
    db_path = f"DataLoader/chroma_db/Visieinfo - {collection_name}"

    # Check if database path exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database path does not exist: {db_path}")
    
    vectorstore = Chroma(
        collection_name=collection_name.replace(" ", "_"), 
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    # Create retriever with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    return retriever

def search_visie_info(query: str, variety_type: Optional[str] = None, top_k: int = 3) -> List[Document]:
    """
    Searches for VisieInfo across all collections or a specific one.

    Args:
        query (str): The search query
        variety_type (str, optional): Specific VisieInfo type to search.
                                     One of: "About", "AI Insights", "Contact", "Documind", "Kothok", "Papers", "Percept", "Verifyid" or "AI solutions for Innovation".
                                     If None, searches all varieties.
        top_k (int): Number of results to return per collection
        
    Returns:
        List[Document]: List of relevant documents
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    results = []
    collections = ["About", "AI Insights", "Contact", "Documind", "Kothok", "Papers", "Percept", "Verifyid", "AI solutions for Innovation"]
    
    if variety_type:
        # Search in specific collection
        try:
            retriever = get_visie_retriever(variety_type, top_k)
            results = retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error searching in collection {variety_type}: {e}")
            return []
    else:
        # Search in all collections
        for collection in collections:
            try:
                retriever = get_visie_retriever(collection, top_k)
                collection_results = retriever.get_relevant_documents(query)
                # Add collection info to metadata
                for doc in collection_results:
                    if hasattr(doc, 'metadata'):
                        doc.metadata['source_collection'] = collection
                    else:
                        doc.metadata = {'source_collection': collection}
                results.extend(collection_results)
            except Exception as e:
                print(f"Error searching in collection {collection}: {e}")
                continue
    
    return results


def get_employees_retriever(top_k: int = 3):
    """
    Creates a retriever for the employees information collection.

    Args:
        top_k (int): Number of results to return
        
    Returns:
        A retriever object that can perform similarity search on the employees information collection.
    """
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Connect to the existing Chroma collection - corrected path
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
    
    try:
        # Get the employees retriever - fixed function name
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
    
    except Exception as e:
        print(f"Error searching employee information: {e}")
        return []



from langchain_core.tools import Tool, tool
from Core.retriver import get_visie_retriever, search_visie_info
from langchain.schema import Document
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_visie_info(query: str, collection_type: Optional[str] = None, top_k: int = 3) -> List[Document]:
    """
    Searches for VisieInfo across all collections or a specific one.

    Args:
        query (str): The search query
        collection_type (str, optional): Specific VisieInfo type to search.
                                     One of: "About", "AI Insights", "Contact", "Documind", "Kothok", "Papers", "Percept", "Verifyid" or "AI solutions for Innovation".
                                     If None, searches all collections.
        top_k (int): Number of results to return per collection
        
    Returns:
        List[Document]: List of relevant documents
    """
    # Input validation
    if not query or not query.strip():
        logging.error("Empty query provided")
        return []
    
    logging.info(f"Searching for query='{query}', collection_type='{collection_type}', top_k={top_k}")

    valid_collections = ["About", "AI Insights", "Contact", "Documind", "Kothok", "Papers", "Percept", "Verifyid", "AI solutions for Innovation"]
    results = []
    
    if collection_type:
        # Validate collection type
        if collection_type not in valid_collections:
            logging.error(f"Invalid collection_type: {collection_type}. Must be one of: {valid_collections}")
            return []
            
        # Search in specific collection
        logging.info(f"Getting retriever for collection: {collection_type}")
        retriever = get_visie_retriever(collection_type, top_k)
        logging.info(f"Retriever obtained. Searching documents for query: '{query}'")
        results = retriever.get_relevant_documents(query)
        logging.info(f"Retrieved {len(results)} documents for '{query}' in '{collection_type}'.")
    else:
        # Search in all collections
        logging.info("No collection_type specified, searching all collections.")
        for visie_type in valid_collections:
            logging.info(f"Getting retriever for collection: {visie_type}")
            retriever = get_visie_retriever(visie_type, top_k)
            logging.info(f"Retriever obtained. Searching documents for query: '{query}' in {visie_type}")
            retrieved_docs = retriever.get_relevant_documents(query)
            # Add collection info to metadata
            for doc in retrieved_docs:
                if hasattr(doc, 'metadata'):
                    doc.metadata['source_collection'] = visie_type
                else:
                    doc.metadata = {'source_collection': visie_type}
            results.extend(retrieved_docs)
            logging.info(f"Retrieved {len(retrieved_docs)} documents for '{query}' in '{visie_type}'.")

    logging.info(f"Total documents found across all searches: {len(results)}")
    return results

def search_visie_info_wrapper(input_str: str) -> str:
    """
    Wrapper function for the LangChain tool that handles string input and returns string output.
    
    Args:
        input_str (str): Input string that can be just a query or "query|collection_type"
        
    Returns:
        str: Formatted search results
    """
    # Parse input - check if collection type is specified
    if "|" in input_str:
        query, collection_type = input_str.split("|", 1)
        query = query.strip()
        collection_type = collection_type.strip() if collection_type.strip() else None
    else:
        query = input_str.strip()
        collection_type = None
    
    # Perform search
    results = search_visie_info(query, collection_type)
    
    if not results:
        return f"No information found for query: '{query}'"
    
    # Format results for output
    formatted_results = []
    for i, doc in enumerate(results[:5], 1):  # Limit to top 5 results
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        source_collection = doc.metadata.get('source_collection', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        formatted_results.append(f"{i}. [{source_collection}] {content}")
    
    return f"Found {len(results)} results for '{query}':\n\n" + "\n\n".join(formatted_results)

@tool
def visie_info_tool(query: str, variety_type: Optional[str] = None) -> str:
    """
    Get information about VISIE.tech projects, products, and services.
    
    Args:
        query: The user's question about VISIE
        variety_type: Optional specific category to search in
    
    Returns:
        Information about VISIE based on the query
    """
    print(f"DEBUG: visie_info_tool called with query: {query}")
    
    # Import and use the actual retriever
    from Core.retriver import search_visie_info
    results = search_visie_info(query, variety_type, top_k=3)
    
    print(f"DEBUG: Retrieved {len(results)} results from database")
    
    if results:
        response_parts = []
        for i, doc in enumerate(results[:3]):  # Limit to top 3 results
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            source = doc.metadata.get('source_collection', 'VISIE Info') if hasattr(doc, 'metadata') else 'VISIE Info'
            
            # Clean and format the content
            clean_content = content.strip()[:500]  # Limit to 500 chars
            response_parts.append(f"**From {source}:**\n{clean_content}")
            
            print(f"DEBUG: Document {i+1} from {source}: {clean_content[:100]}...")
        
        final_response = "\n\n".join(response_parts)
        print(f"DEBUG: Final response length: {len(final_response)}")
        return final_response
    else:
        fallback_response = f"I searched for information about '{query}' in VISIE's knowledge base. While I couldn't find specific details in our database, I can tell you that VISIE is an innovative AI technology company offering:\n\n• **Documind**: AI-powered document processing and analysis\n• **Kothok**: Advanced conversational AI and chatbot solutions\n• **Percept**: Computer vision and image recognition technology\n• **VerifyID**: Identity verification and authentication systems\n• **AI Solutions for Innovation**: Custom AI solutions for businesses\n\nCould you please be more specific about what aspect of VISIE you'd like to know about?"
        
        print("DEBUG: No results found, returning fallback response")
        return fallback_response

visie_info_tool = Tool(
    name="visie_info_tool",
    func=search_visie_info_wrapper,
    description="Retrieves detailed information about Visie from various collections. Input format: 'query' or 'query|collection_type'. Available collections: About, AI Insights, Contact, Documind, Kothok, Papers, Percept, Verifyid, AI solutions for Innovation. Use this tool to find information about Visie company, products, services, team members, research papers, and AI solutions."
)
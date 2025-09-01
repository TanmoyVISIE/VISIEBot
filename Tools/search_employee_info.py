from langchain_core.tools import Tool, tool
from Core.retriver import search_employees_info
from langchain.schema import Document
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_employee_info_wrapper(input_str: str) -> str:
    """
    Wrapper function for the LangChain tool that handles string input and returns string output.
    
    Args:
        input_str (str): Input string containing the search query for employee information and their skills.

    Returns:
        str: Formatted search results about VISIE employees
    """
    query = input_str.strip()
    
    if not query:
        return "Please provide a search query for employee information."
    
    # Perform search
    results = search_employees_info(query)
    
    if not results:
        return f"No employee information found for query: '{query}'"
    
    # Format results for output
    formatted_results = []
    for i, doc in enumerate(results[:3], 1):  # Limit to top 3 results
        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        formatted_results.append(f"{i}. {content}")
    
    return f"Found {len(results)} employee results for '{query}':\n\n" + "\n\n".join(formatted_results)

@tool
def visie_employee_tool(query: str) -> str:
    """
    Get information about VISIE employees, team members, and staff details.
    
    Args:
        query: The user's question about VISIE employees or team members
    
    Returns:
        Information about VISIE employees based on the query
    """
    print(f"DEBUG: visie_employee_tool called with query: {query}")
    
    # Use the employee information retriever
    results = search_employees_info(query, top_k=3)
    
    print(f"DEBUG: Retrieved {len(results)} employee results from database")
    
    if results:
        response_parts = []
        for i, doc in enumerate(results[:3]):  # Limit to top 3 results
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            # Clean and format the content
            clean_content = content.strip()[:400]  # Limit to 400 chars
            response_parts.append(f"**Employee Info {i+1}:**\n{clean_content}")
            
            print(f"DEBUG: Employee document {i+1}: {clean_content[:100]}...")
        
        final_response = "\n\n".join(response_parts)
        print(f"DEBUG: Final employee response length: {len(final_response)}")
        return final_response
    else:
        fallback_response = f"I searched for employee information about '{query}' in VISIE's team database. While I couldn't find specific details, I can help you with information about VISIE's team structure and employee-related queries. Could you please be more specific about which team member or department you're looking for?"
        
        print("DEBUG: No employee results found, returning fallback response")
        return fallback_response

# Create the tool instance
visie_employee_tool_instance = Tool(
    name="visie_employee_tool",
    func=search_employee_info_wrapper,
    description="Retrieves information about VISIE employees, team members, staff details, and organizational structure. Use this tool to find information about specific employees, departments, roles, or team composition at VISIE."
)
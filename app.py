import os
from typing import Optional, TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
from Tools.search_visie_info import visie_info_tool
# from Tools.weather import weather_info_tool
from Tools.search_tool import search
from Core.prompt import SYSTEM
from langchain_google_vertexai import GemmaLocalHF, GemmaChatLocalHF
from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import base64
import io
import json
import traceback

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="VISIE RAG Chatbot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="VISIEBot"), name="static")
templates = Jinja2Templates(directory="VISIEBot")

# # Create uploads directory if it doesn't exist
# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Setup LLM and history
# # Get API key from environment variable
# google_api_key = os.environ.get("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("GOOGLE_API_KEY environment variable not set")

# Get API key from environment variable
hf_access_token = os.environ.get("HF_TOKEN")
if not hf_access_token:
    raise ValueError("HF_TOKEN environment variable not set")

# Initialize LLM
llm = GemmaLocalHF(
    model_name="google/gemma-3-270m", 
    temperature=0,
    hf_access_token=hf_access_token
)

# History management
HISTORY_FILE = "history.txt"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as file:
        file.write("")

def read_previous_history():
    """Read the previous_response_id from the history file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    return None

def write_previous_history(messages):
    """Append the messages to the history file."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as file:
        if isinstance(messages, list):
            message_str = str(messages)
        else:
            message_str = str(messages)
        file.write(message_str + "\n")

# Define the prompt template for the main agent interaction
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM.format(context="")),
        ("placeholder", "{messages}"),
    ]
)

# Tools setup
tools = [
    visie_info_tool,
    search
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)
agent_chain = agent_prompt | llm_with_tools

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    response = agent_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

# Build the graph
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

def get_final_response(messages):
    """Extract the final response from the list of messages."""
    try:
        previous_messages = read_previous_history()
        if previous_messages:
            messages = [HumanMessage(content=previous_messages)] + messages
        
        response = alfred.invoke({"messages": messages})
        final_response = next(
            (m for m in reversed(response['messages']) 
             if isinstance(m, AIMessage) and not m.tool_calls), 
            None
        )
        
        if final_response:
            write_previous_history(final_response.content)
            print("Alfred's Response:", final_response)
            return final_response.content
        else:
            return "I apologize, but I couldn't generate a proper response. Please try again."
            
    except Exception as e:
        print(f"Error in get_final_response: {e}")
        traceback.print_exc()
        return f"An error occurred while processing your request: {str(e)}"

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """Handle chat requests from the frontend."""
    try:
        if not chat_request.message or not chat_request.message.strip():
            raise HTTPException(status_code=400, detail="No message provided")
        
        print(f"User Input: {chat_request.message}")
        
        # Create human message
        human_message = HumanMessage(content=chat_request.message.strip())
        
        # Get response from the agent
        response_content = get_final_response([human_message])
        
        print(f"Bot Response: {response_content}")
        
        return ChatResponse(
            response=response_content,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "VISIE RAG Chatbot"}

@app.post("/clear-history")
async def clear_history():
    """Clear chat history."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w") as file:
                file.write("")
        return {"status": "success", "message": "History cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True,
        log_level="info"
    )
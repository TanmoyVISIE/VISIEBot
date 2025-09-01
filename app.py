import traceback
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI, HTTPException, Request
from langchain_core.runnables import RunnableLambda
from Core.prompt import SYSTEM
from Tools.search_tool import search
from Tools.search_visie_info import visie_info_tool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
import os
from typing import Optional, TypedDict, Annotated
from dotenv import load_dotenv

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
app.mount("/VISIEBot", StaticFiles(directory="VISIEBot"), name="static")
templates = Jinja2Templates(directory="VISIEBot")

# Get API keys from environment variables
google_api_key = os.environ.get("GOOGLE_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

if not google_api_key:
    print("Warning: GOOGLE_API_KEY not found. Will try to use Hugging Face models only.")

# Initialize LLM with multiple fallback options
llm = None

# First try: Gemini 1.5 Flash
if google_api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=google_api_key
    )
    print("Successfully initialized Google Gemini 1.5 Flash model")

# Second fallback: DialoGPT
if not llm:
    print("Attempting to initialize Microsoft DialoGPT model...")
    
    model_name = "microsoft/DialoGPT-medium"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if hf_token else None,
        padding_side='left'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token if hf_token else None,
        torch_dtype="auto",
        device_map="cpu"
    )

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create pipeline with simpler settings
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        device_map="cpu"
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("Successfully initialized Microsoft DialoGPT model")

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

# Define simple system prompt as fallback
SIMPLE_SYSTEM = "You are VISIEBot, an AI assistant that helps users learn about VISIE AI, its products, services, and solutions. Use the available tools when users ask about VISIE-related topics."

# Define the prompt template for the main agent interaction
def create_agent_prompt():
    test_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("placeholder", "{messages}"),
    ])
    
    # Check if it needs context
    if 'context' in test_prompt.input_variables:
        print("DEBUG: Original prompt requires context, providing it...")
        def format_with_context(input_data):
            return {
                "messages": input_data["messages"],
                "context": ""
            }
        return RunnableLambda(format_with_context) | test_prompt
    else:
        print("DEBUG: Using original prompt without context")
        return test_prompt

agent_prompt = create_agent_prompt()

# Tools setup
tools = [
    visie_info_tool,
    search
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Create agent chain with context handling
def create_agent_with_context():
    def format_with_context(input_data):
        return {
            "messages": input_data["messages"],
            "context": ""
        }
    
    return RunnableLambda(format_with_context) | agent_prompt | llm_with_tools

agent_chain = create_agent_with_context()

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
    print(f"DEBUG: Processing message: {[m.content if hasattr(m, 'content') else str(m) for m in messages]}")
    
    print("DEBUG: Invoking agent...")
    response = alfred.invoke({"messages": messages})
    print(f"DEBUG: Agent response: {response}")
    
    final_response = next(
        (m for m in reversed(response['messages'])
         if isinstance(m, AIMessage) and not m.tool_calls),
        None
    )

    if final_response:
        write_previous_history(final_response.content)
        print("Alfred's Response:", final_response.content)
        return final_response.content
    else:
        print("DEBUG: No final response found in agent output")
        # Return the last AI message even if it has tool calls
        last_ai_message = next(
            (m for m in reversed(response['messages']) if isinstance(m, AIMessage)),
            None
        )
        if last_ai_message:
            return last_ai_message.content
        return "I apologize, but I couldn't generate a proper response. Please try again."

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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "VISIE RAG Chatbot"}

@app.post("/clear-history")
async def clear_history():
    """Clear chat history."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as file:
            file.write("")
    return {"status": "success", "message": "History cleared"}

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

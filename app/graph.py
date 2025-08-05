from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, END
from . import services

# --- State Definition for Interactive Graph ---
class InteractiveGraphState(TypedDict):
    user_prompt: str
    chat_id: int
    intent: NotRequired[str]
    entity: NotRequired[str] # Will hold the extracted name, e.g., "google"
    ticker: NotRequired[str] # Will hold the final ticker, e.g., "GOOGL"
    price_data: NotRequired[dict]
    news_data: NotRequired[list[str]]
    analysis: NotRequired[str]

# --- Node Functions ---
async def classify_intent_node(state: InteractiveGraphState) -> dict:
    intent = await services.classify_intent(state['user_prompt'])
    return {"intent": intent}

async def extract_entity_node(state: InteractiveGraphState) -> dict:
    entity = await services.extract_entity(state['user_prompt'])
    return {"entity": entity}

async def map_entity_to_ticker_node(state: InteractiveGraphState) -> dict:
    entity = state.get('entity')
    if not entity:
        return {"ticker": None}
    ticker = await services.get_ticker_from_entity(entity)
    return {"ticker": ticker}

async def fetch_data_node(state: InteractiveGraphState) -> dict:
    ticker = state.get('ticker')
    if not ticker:
        return {"price_data": {}, "news_data": [], "analysis": "I could not identify a valid stock ticker in your request. Please try again with a company name like 'Apple' or a ticker like 'AAPL'."}
    
    price_data = await services.get_stock_data_interactive(ticker)
    if not price_data:
        # If price fetch fails, we can't proceed.
        return {"price_data": {}, "news_data": [], "analysis": f"Sorry, I could not retrieve price data for {ticker}. The ticker might be invalid or the data source is unavailable."}
        
    news_data = await services.fetch_stock_news_interactive(ticker)
    return {"price_data": price_data, "news_data": news_data}

async def generate_analysis_node(state: InteractiveGraphState) -> dict:
    """Generates the final AI analysis."""
    # Check if a pre-made analysis/error message already exists from a previous node
    if state.get("analysis"):
        return {} # Passthrough if an error message was already generated
        
    ticker = state.get('ticker')
    price_data = state.get('price_data')
    news_data = state.get('news_data') or []
    
    # Combine the checks for both ticker and price_data
    if not ticker or not price_data:
        return {"analysis": "An error occurred: Ticker or price data was not found."}

    analysis = await services.generate_single_stock_analysis(ticker, price_data, news_data)
    return {"analysis": analysis}

async def send_response_node(state: InteractiveGraphState) -> dict:
    """Sends the final analysis or an error message to the user."""
    chat_id = state['chat_id']
    analysis = state.get('analysis', "An unexpected error occurred.")
    await services.send_telegram_message(chat_id, analysis)
    return {}

async def send_rejection_node(state: InteractiveGraphState) -> dict:
    chat_id = state['chat_id']
    message = "I'm sorry, I can only provide analysis for specific stocks and companies."
    await services.send_telegram_message(chat_id, message)
    return {}

# --- Conditional Logic ---
def route_by_intent(state: InteractiveGraphState) -> str:
    intent = state.get('intent', 'OTHER')
    return "analyze_stock" if intent == 'STOCK' else "reject_prompt"

# --- Graph Definition ---
def create_interactive_graph():
    workflow = StateGraph(InteractiveGraphState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("extract_entity", extract_entity_node)
    workflow.add_node("map_entity_to_ticker", map_entity_to_ticker_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("generate_analysis", generate_analysis_node)
    workflow.add_node("send_response", send_response_node)
    workflow.add_node("send_rejection", send_rejection_node)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent", route_by_intent,
        {"analyze_stock": "extract_entity", "reject_prompt": "send_rejection"}
    )
    
    workflow.add_edge("extract_entity", "map_entity_to_ticker")
    workflow.add_edge("map_entity_to_ticker", "fetch_data")
    workflow.add_edge("fetch_data", "generate_analysis")
    workflow.add_edge("generate_analysis", "send_response")
    workflow.add_edge("send_response", END)
    workflow.add_edge("send_rejection", END)
    
    return workflow.compile()

# --- Graph Initialization ---
interactive_graph_agent = create_interactive_graph()
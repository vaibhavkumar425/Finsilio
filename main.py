import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.graph import interactive_graph_agent

# Defines the structure of the incoming data from Telegram
class Chat(BaseModel):
    id: int

class Message(BaseModel):
    message_id: int
    chat: Chat
    text: Optional[str] = None

class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[Message] = None

# --- App Initialization ---
app = FastAPI(
    title="Finsilio Bot",
    description="An interactive stock analysis bot using LangGraph and FastAPI.",
    version="2.0.0",
)

# Load environment variables from .env file
load_dotenv()

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def root():
    """Root endpoint to confirm the server is running."""
    return {"message": "Finsilio Interactive Bot is running!"}

@app.post("/telegram-webhook", tags=["Telegram"])
async def telegram_webhook(update: TelegramUpdate):
    """This endpoint receives messages from Telegram."""
    if update.message and update.message.text:
        user_prompt = update.message.text
        chat_id = update.message.chat.id
        print(f"Received message from chat_id {chat_id}: {user_prompt}")
        
        # Call our interactive graph agent with the user's prompt and chat ID
        await interactive_graph_agent.ainvoke({
            "user_prompt": user_prompt,
            "chat_id": chat_id
        })
        
    return {"status": "ok"}

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
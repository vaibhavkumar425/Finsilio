import os
import yfinance as yf
from telegram import Bot
from fastapi.concurrency import run_in_threadpool
import json
from google import genai

async def send_telegram_message(chat_id: int, message: str):
    """Sends a message to a specific Telegram chat ID."""
    print(f"Final message is: {message}")
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: Telegram token not found.")
        return
    try:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message)
        print(f"Successfully sent message to chat_id {chat_id}.")
    except Exception as e:
        print(f"Error sending message to chat_id {chat_id}: {e}")

async def classify_intent(user_prompt: str) -> str:
    """Uses the LLM to classify the user's intent."""
    print("--- Classifying Intent ---")
    if not os.getenv("GOOGLE_API_KEY"):
        return "ERROR"
    try:
        client = genai.Client()
        prompt = f"""Is the following user prompt asking for financial analysis or data about a specific company or stock? 
        Answer ONLY with the word 'STOCK' or 'OTHER'.

        User Prompt: "{user_prompt}"
        """
        response = await client.aio.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        text_response = response.text or ""
        classification = text_response.strip().upper()
        print(f"Intent classified as: {classification}")
        return "STOCK" if "STOCK" in classification else "OTHER"
    except Exception as e:
        print(f"Error classifying intent: {e}")
        return "OTHER"

async def extract_entity(user_prompt: str) -> str | None:
    """Uses the LLM to extract a single company name or stock ticker from a prompt."""
    print("--- Extracting Entity ---")
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        client = genai.Client()
        prompt = f"""From the following user prompt, extract the single most likely company name or stock ticker.
        Return ONLY the company name or stock ticker. If no specific company or ticker is mentioned, return 'NONE'.

        User Prompt: "{user_prompt}"
        """
        response = await client.aio.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        entity = (response.text or "").strip()
        print(f"Extracted entity: {entity}")
        return None if "NONE" in entity.upper() else entity
    except Exception as e:
        print(f"Error extracting entity: {e}")
        return None


async def get_ticker_from_entity(entity: str) -> str | None:
    """Uses the LLM to find the official stock ticker for a given company entity."""
    print(f"--- Mapping Entity '{entity}' to Ticker ---")
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        client = genai.Client()
        prompt = f"""What is the official stock ticker for the company '{entity}'? 
        Return ONLY the ticker symbol (e.g., GOOGL, AAPL). 
        If you cannot find a ticker, return 'NONE'.
        """
        response = await client.aio.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        ticker = (response.text or "").strip().upper()
        print(f"Mapped to ticker: {ticker}")
        return None if "NONE" in ticker else ticker
    except Exception as e:
        print(f"Error mapping entity to ticker: {e}")
        return None


def _fetch_yf_data_interactive(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        print("stock_info ",stock.info)
        return dict(stock.fast_info)
    except Exception as e:
        print(f"Could not fetch price data for {ticker}: {e}")
        return {}
async def get_stock_data_interactive(ticker: str) -> dict:
    print(f"--- Fetching Price Data for {ticker} ---")
    return await run_in_threadpool(_fetch_yf_data_interactive, ticker)
async def fetch_stock_news_interactive(ticker: str) -> list[str]:
    print(f"--- Fetching News for {ticker} ---")
    try:
        headlines = await run_in_threadpool(lambda: yf.Ticker(ticker).news)
        titles = [item.get('content', {}).get('title') for item in headlines[:5] if item.get('content', {}).get('title')]
        return titles
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []
    

async def generate_single_stock_analysis(ticker: str, price_data: dict, news_data: list[str]) -> str:
    """
    Generates a focused analysis for a single stock using the Gemini LLM.
    """
    print(f"--- Generating AI Analysis for {ticker} ---")
    print("--- Raw Price Data Received by AI Service ---")
    print(price_data)
    print("-------------------------------------------")
    # ------------------------------------

    if not os.getenv("GOOGLE_API_KEY"):
        return "Error: Google API Key not configured."
    
    relevant_price_data = {
        "last_price": price_data.get('lastPrice'),
        "previous_close": price_data.get('previousClose'),
        "day_high": price_data.get('dayHigh'),
        "day_low": price_data.get('dayLow'),
        "52_week_high": price_data.get('yearHigh'),
        "52_week_low": price_data.get('yearLow'),
        "market_cap": price_data.get('marketCap'),
    }

    try:
        client = genai.Client()
        prompt = f"""
        You are a financial analyst for 'Finsilio'. Provide a concise, professional analysis for the stock: {ticker}.

        Use the following data to form your analysis. If data is missing or empty, state that you couldn't retrieve it.
        - Price Data: {json.dumps(relevant_price_data)}
        - Recent News Headlines: {json.dumps(news_data)}

        Structure your response in Markdown with the following sections:
        - A brief, one-paragraph **Summary** of the stock's current situation.
        - Key **Data Points** in a bulleted list.
        - A short **News Sentiment** section (Positive, Negative, or Neutral) based on the headlines.
        """
        response = await client.aio.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text or "Error: The AI model returned no content."
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        return "Error: Failed to generate AI analysis."
import re
import sys
import pytz
import urllib3
import certifi
import requests
import json
from bs4 import BeautifulSoup
from time import perf_counter as pre
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# LangChain & Ollama Imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

MAX_WORKERS = 5
BD_TIMEZONE = pytz.timezone('Asia/Dhaka')

HEADERS = {'accept': '*/*', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0'}
URL_TEMPLATES = {
    'DSE': 'https://www.dsebd.org/displayCompany.php?name={symbol}',
    'CSE': 'https://www.cse.com.bd/company/companydetails/{symbol}',
    'SME': 'https://sme.dsebd.org/sme_displayCompany.php?name={symbol}'
}

session = requests.Session()
session.headers.update(HEADERS)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_soup(url, verify=True):
    try:
        response = session.get(url, timeout=15, verify=verify)
        response.raise_for_status()
        
        try:
            return BeautifulSoup(response.content, 'lxml')
        except Exception:
            return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"[✗] Error fetching {url}: {e}")
        return None

def process_with_rag(table_html):
    try:
        # 1. Setup Embeddings and Model
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        llm = ChatOllama(model="llama3.2", temperature=0)
        # llm = ChatOllama(model="deepseek-r1:latest", temperature=0)

        # 2. Store entire table as single document to preserve structure
        vectorstore = FAISS.from_texts([table_html], embedding=embeddings)
        
        # 3. Retrieve the full table (single doc, so k=1)
        retrieved_docs = vectorstore.similarity_search("Share Holding Percentage", k=1)
        context = retrieved_docs[0].page_content

        # 4. Prompt Engineering for extraction
        today = datetime.now(BD_TIMEZONE).strftime("%d%m%Y")
        template = """You are a strict JSON data extractor. Your job is to extract ALL 'Share Holding Percentage' rows from the HTML context below.

        Context:
        {context}

        Rules:
        1. Extract EVERY "Share Holding Percentage" row. There may be multiple rows (e.g., different dates). Extract ALL of them.
        2. Convert dates to DDMMYYYY integer format. Example: "Jun 30, 2025" becomes 30062025, "Feb 28, 2026" becomes 28022026, "Mar 31, 2025" becomes 31032025.
        3. Use these exact keys: holding_date, sponsor_director, govt_share, institute_share, foreign_share, public_share, last_update.
        4. All numeric values must be floats (e.g., 0.00 not "0.00").
        5. holding_date must be an integer (e.g., 30062025 not "30062025").
        6. last_update for all rows must be: "{today}"
        7. Return ONLY a valid JSON array. No explanation. No markdown. No code blocks. Just the raw JSON array.

        Example output format (with PLACEHOLDER values, do NOT copy these numbers):
        [{{{{
            "holding_date": 15011999,
            "sponsor_director": 11.11,
            "govt_share": 22.22,
            "institute_share": 33.33,
            "foreign_share": 44.44,
            "public_share": 55.55,
            "last_update": "{today}"
        }}}}]"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm

        # 5. Invoke 
        response = chain.invoke({"context": context, "today": today})
        return response.content

        # this response proccess for deepseek chat model
        # cleaned = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        # return cleaned
    except Exception as e:
        return f"Error in RAG process: {e}"


def fetch_scrap_data(symbol_data):
    symbol, exchange = symbol_data
    url = URL_TEMPLATES.get(exchange).format(symbol=symbol)
    
    verify = certifi.where() if exchange != 'CSE' else False
    
    soup = get_soup(url, verify=verify)
    if not soup: return None

    list_tables = soup.find_all('table', {'id': 'company'})
    try:
        if exchange == 'SME' and len(list_tables) > 9:
            holding_table = list_tables[9]
        elif len(list_tables) > 10:
            holding_table = list_tables[10]
        else:
            # Fallback: find it by text if the list is shorter than expected
            holding_table = next((t for t in list_tables if "Other Information" in t.text), None)
    except IndexError:
        holding_table = None

    if holding_table:
        print(f"[✓] Processing {symbol} with Local LLM (RAG)...")

        # Pass raw HTML to preserve structure (dates + values stay together)
        table_html = str(holding_table)
        json_output = process_with_rag(table_html)
        
        print(f"\nHolding Data for {symbol}:\n{json_output}\n")
        return json_output
    
    print(f"[✗] Could not find holding table for {symbol}")
    return None
    

def main(symbol=None, exchange=None):
    start = pre()

    if symbol is None or exchange is None:
        symbols = None # default symbols and exchange comes from db
    else:
        symbols = [(symbol.upper(), exchange.upper())]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(fetch_scrap_data, symbols))
    
    end = pre()
    print(f"--- Completed in {end - start:.2f} seconds ---")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        symbol, exchange = sys.argv[1], sys.argv[2]
        main(symbol, exchange)
    else:
        main()
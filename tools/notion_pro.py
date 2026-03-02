import os
import requests
from typing import List, Dict, Any

NOTION_TOKEN = os.getenv("NOTION_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def search_notion_by_title(title: str) -> List[Dict[str, Any]]:
    """Searches for Notion pages by title dynamically."""
    url = "https://api.notion.com/v1/search"
    payload = {"query": title, "filter": {"value": "page", "property": "object"}}
    response = requests.post(url, json=payload, headers=HEADERS)
    return response.json().get("results", [])

def get_database_schema(database_id: str) -> Dict[str, Any]:
    """Retrieves properties of a specific database to prevent errors."""
    url = f"https://api.notion.com/v1/databases/{database_id}"
    response = requests.get(url, headers=HEADERS)
    return response.json().get("properties", {})

import requests
from typing import Dict

from .pubmed_search import PubmedSearchAgent
from .tavily_search import TavilySearchAgent

class WebSearchAgent:
    def __init__(self, config):
        self.tavily_search_agent = TavilySearchAgent()
    
    def search(self, query: str) -> str:
        tavily_results = self.tavily_search_agent.search_tavily(query=query)
        return f"Tavily Results:\n{tavily_results}\n"

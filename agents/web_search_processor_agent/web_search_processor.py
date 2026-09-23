import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """
        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        
        web_results = self.web_search_agent.search(web_search_query.content)
        
        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )
        
        response = self.llm.invoke(llm_prompt)
        return response

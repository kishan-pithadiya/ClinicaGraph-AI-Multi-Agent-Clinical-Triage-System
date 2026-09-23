"""
ClinicaGraph AI - Multi-Provider LLM & Embeddings Orchestration Engine
Supports:
- OpenAI (Direct API: gpt-4o, gpt-4o-mini, text-embedding-3-small)
- Azure OpenAI (Enterprise deployments)
- Fallback / Offline Mock provider for testing and deterministic validation
"""

import os
import logging
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("ClinicaGraph.LLMProvider")


class MockChatModel:
    """Fallback simulated LLM provider for demonstration and offline testing."""
    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature

    def invoke(self, prompt: Any) -> Any:
        from langchain_core.messages import AIMessage
        
        prompt_str = str(prompt).lower()
        
        # Decision agent routing response
        if "which agent should handle" in prompt_str or "lead clinical triage supervisor" in prompt_str:
            if "mri" in prompt_str or "brain" in prompt_str:
                return AIMessage(content='{"agent": "BRAIN_TUMOR_AGENT", "urgency": "ROUTINE", "reasoning": "Detected brain MRI imaging context", "confidence": 0.95}')
            elif "chest" in prompt_str or "x-ray" in prompt_str or "covid" in prompt_str:
                return AIMessage(content='{"agent": "CHEST_XRAY_AGENT", "urgency": "ROUTINE", "reasoning": "Detected chest radiography context", "confidence": 0.95}')
            elif "skin" in prompt_str or "lesion" in prompt_str or "rash" in prompt_str:
                return AIMessage(content='{"agent": "SKIN_LESION_AGENT", "urgency": "ROUTINE", "reasoning": "Detected dermatological examination query", "confidence": 0.95}')
            elif "latest" in prompt_str or "outbreak" in prompt_str or "recent" in prompt_str or "news" in prompt_str:
                return AIMessage(content='{"agent": "WEB_SEARCH_PROCESSOR_AGENT", "urgency": "ROUTINE", "reasoning": "Query requires real-time medical updates", "confidence": 0.92}')
            elif "treatment" in prompt_str or "symptoms" in prompt_str or "protocol" in prompt_str:
                return AIMessage(content='{"agent": "RAG_AGENT", "urgency": "ROUTINE", "reasoning": "Medical literature lookup required", "confidence": 0.90}')
            else:
                return AIMessage(content='{"agent": "CONVERSATION_AGENT", "urgency": "INFORMATIONAL", "reasoning": "Standard clinical consultation interaction", "confidence": 0.95}')
        
        # Guardrail prompt response
        if "content safety filter" in prompt_str:
            if any(term in prompt_str for term in ["suicide", "harm myself", "lethal dose", "make a bomb", "cyanide"]):
                return "UNSAFE: Harmful or self-harm content detected. Please contact emergency services immediately (988 / 911)."
            return "SAFE"

        # Vision classification response
        if "classify it as" in prompt_str:
            return AIMessage(content='{"image_type": "CHEST X-RAY", "reasoning": "Visual characteristics indicate pulmonary radiography", "confidence": 0.92}')

        # Default clinical response
        return AIMessage(
            content=(
                "**[ClinicaGraph Clinical Insight]**\n\n"
                "Based on clinical knowledge guidelines, symptoms should be evaluated in accordance "
                "with standard diagnostic criteria. If experiencing persistent, severe, or worsening "
                "symptoms, immediate consultation with a board-certified healthcare provider is recommended.\n\n"
                "*Disclaimer: ClinicaGraph is an AI clinical decision support system and does not replace formal clinical judgement.*"
            )
        )

    def __or__(self, other):
        from langchain_core.runnables import RunnableSequence
        return RunnableSequence(self, other)


class MockEmbeddings:
    """Fallback simulated embedding generator generating deterministic 1536-dim vectors."""
    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)
        import random
        rng = random.Random(seed)
        return [rng.uniform(-0.1, 0.1) for _ in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def get_llm(temperature: float = 0.2, streaming: bool = False):
    """
    Dynamically instantiate the appropriate LLM based on environment configuration.
    Priority:
    1. Azure OpenAI (if deployment_name and azure_endpoint are provided)
    2. Direct OpenAI (if OPENAI_API_KEY is present)
    3. MockChatModel (fallback mode for offline execution/demos)
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()

    # 1. Check Azure OpenAI
    if provider == "azure" or (
        provider == "auto" 
        and os.getenv("azure_endpoint") 
        and os.getenv("openai_api_key")
    ):
        try:
            from langchain_openai import AzureChatOpenAI
            return AzureChatOpenAI(
                deployment_name=os.getenv("deployment_name", "gpt-4o"),
                model_name=os.getenv("model_name", "gpt-4o"),
                azure_endpoint=os.getenv("azure_endpoint"),
                openai_api_key=os.getenv("openai_api_key"),
                openai_api_version=os.getenv("openai_api_version", "2024-02-15-preview"),
                temperature=temperature,
                streaming=streaming
            )
        except Exception as e:
            logger.warning(f"Could not initialize AzureChatOpenAI: {e}. Falling back...")

    # 2. Check Direct OpenAI
    openai_key = os.getenv("OPENAI_API_KEY") or (
        os.getenv("openai_api_key") if not os.getenv("azure_endpoint") else None
    )
    if provider == "openai" or (provider == "auto" and openai_key):
        try:
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
            return ChatOpenAI(
                model=model_name,
                api_key=openai_key,
                temperature=temperature,
                streaming=streaming
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatOpenAI: {e}. Falling back...")

    # 3. Fallback / Mock
    logger.info("Initializing ClinicaGraph Offline/Mock Chat Provider.")
    return MockChatModel(temperature=temperature)


def get_embeddings():
    """
    Dynamically instantiate embeddings model based on environment configuration.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()

    # 1. Check Azure
    if provider == "azure" or (
        provider == "auto" 
        and os.getenv("embedding_azure_endpoint") 
        and os.getenv("embedding_openai_api_key")
    ):
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            return AzureOpenAIEmbeddings(
                deployment=os.getenv("embedding_deployment_name", "text-embedding-3-small"),
                model=os.getenv("embedding_model_name", "text-embedding-3-small"),
                azure_endpoint=os.getenv("embedding_azure_endpoint"),
                openai_api_key=os.getenv("embedding_openai_api_key"),
                openai_api_version=os.getenv("embedding_openai_api_version", "2024-02-15-preview")
            )
        except Exception as e:
            logger.warning(f"Could not initialize AzureOpenAIEmbeddings: {e}. Falling back...")

    # 2. Direct OpenAI
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if provider == "openai" or (provider == "auto" and openai_key):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=openai_key
            )
        except Exception as e:
            logger.warning(f"Could not initialize OpenAIEmbeddings: {e}. Falling back...")

    # 3. Fallback Mock Embeddings
    logger.info("Initializing ClinicaGraph Offline/Mock Embeddings Provider.")
    return MockEmbeddings()



"""
Configuration file for ClinicaGraph AI: Multi-Agent Clinical Triage System

Centralizes configuration for agent orchestration, RAG vector retrieval,
computer vision models, safety guardrails, speech services, and API server settings.
"""

import os
from dotenv import load_dotenv
from core.llm_provider import get_llm, get_embeddings

# Load environment variables
load_dotenv()


class AgentDecisionConfig:
    """Configuration for the supervisor routing & clinical triage agent."""
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.confidence_threshold = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.85"))
        self.enable_urgency_triage = True


# Backwards compatibility alias
AgentDecisoinConfig = AgentDecisionConfig


class ConversationConfig:
    """Configuration for general clinical conversation agent."""
    def __init__(self):
        self.llm = get_llm(temperature=0.5)


class WebSearchConfig:
    """Configuration for real-time PubMed & clinical literature web search."""
    def __init__(self):
        self.llm = get_llm(temperature=0.2)
        self.context_limit = int(os.getenv("WEB_SEARCH_CONTEXT_LIMIT", "10"))
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")


class RAGConfig:
    """Configuration for Qdrant-backed Retrieval-Augmented Generation."""
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
        self.distance_metric = "Cosine"
        self.use_local = os.getenv("QDRANT_USE_LOCAL", "true").lower() == "true"
        self.vector_local_path = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant_db")
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "clinicagraph_clinical_rag")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # Models
        self.embedding_model = get_embeddings()
        self.llm = get_llm(temperature=0.2)
        self.summarizer_model = get_llm(temperature=0.3)
        self.chunker_model = get_llm(temperature=0.0)
        self.response_generator_model = get_llm(temperature=0.2)
        
        # Retrieval Tuning
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.vector_search_type = 'similarity'
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = int(os.getenv("RERANKER_TOP_K", "3"))
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "8192"))
        self.include_sources = True
        self.min_retrieval_confidence = float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", "0.40"))
        self.context_limit = 10


class MedicalCVConfig:
    """Configuration for multimodal imaging agents (Brain MRI, Chest X-ray, Skin Lesion)."""
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.brain_tumor_output_path = "./uploads/brain_tumor_output/mri_segmentation.png"
        self.llm = get_llm(temperature=0.1)


class SpeechConfig:
    """Configuration for voice dictation & speech synthesis."""
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = os.getenv("ELEVEN_LABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")


class ValidationConfig:
    """Configuration for clinician human-in-the-loop review."""
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"


class APIConfig:
    """Configuration for FastAPI web server."""
    def __init__(self):
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.rate_limit = int(os.getenv("RATE_LIMIT", "60"))
        self.max_image_upload_size = int(os.getenv("MAX_UPLOAD_MB", "10"))


class UIConfig:
    """Configuration for front-end appearance and telemetry."""
    def __init__(self):
        self.app_title = "ClinicaGraph AI"
        self.app_subtitle = "Autonomous Multi-Agent Clinical Triage System"
        self.theme = "clinical-light"
        self.enable_speech = True
        self.enable_image_upload = True
        self.enable_soap_export = True


class Config:
    """Unified application configuration root."""
    def __init__(self):
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = int(os.getenv("MAX_HISTORY", "20"))


# Global singleton instance
config = Config()
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
from langchain_core.runnables import Runnable
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

logger = logging.getLogger("ClinicaGraph.LLMProvider")


class MockChatModel(Runnable):
    """Fallback simulated LLM provider for demonstration and offline testing."""
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs) -> Any:
        prompt_str = str(input).lower()
        
        # 1. Lead Clinical Triage Supervisor Router
        if "which agent should handle" in prompt_str or "lead clinical triage supervisor" in prompt_str or "supervisor_system_prompt" in prompt_str:
            query_part = prompt_str.split("query:")[-1].split("has image:")[0].strip() if "query:" in prompt_str else prompt_str
            has_image = "has image: true" in prompt_str

            if has_image:
                if "brain" in prompt_str or "mri" in prompt_str:
                    return AIMessage(content='{"agent": "BRAIN_TUMOR_AGENT", "urgency": "ROUTINE", "reasoning": "Detected brain MRI image attachment", "confidence": 0.96}')
                elif "skin" in prompt_str or "lesion" in prompt_str or "rash" in prompt_str:
                    return AIMessage(content='{"agent": "SKIN_LESION_AGENT", "urgency": "ROUTINE", "reasoning": "Detected dermatological image attachment", "confidence": 0.95}')
                else:
                    return AIMessage(content='{"agent": "CHEST_XRAY_AGENT", "urgency": "ROUTINE", "reasoning": "Detected chest radiograph attachment", "confidence": 0.95}')
            
            # Text queries:
            if any(term in query_part for term in ["brain tumor", "glioma", "meningioma"]):
                return AIMessage(content='{"agent": "RAG_AGENT", "urgency": "ROUTINE", "reasoning": "Clinical literature retrieval required for neuro-oncology", "confidence": 0.92}')
            elif any(term in query_part for term in ["outbreak", "covid stats", "latest research", "new drug", "news"]):
                return AIMessage(content='{"agent": "WEB_SEARCH_PROCESSOR_AGENT", "urgency": "ROUTINE", "reasoning": "Query requires real-time web surveillance", "confidence": 0.92}')
            elif any(term in query_part for term in ["treatment protocol", "medical paper", "clinical guideline", "pneumonia research"]):
                return AIMessage(content='{"agent": "RAG_AGENT", "urgency": "ROUTINE", "reasoning": "Clinical literature retrieval required", "confidence": 0.92}')
            else:
                return AIMessage(content='{"agent": "CONVERSATION_AGENT", "urgency": "INFORMATIONAL", "reasoning": "Standard clinical consultation interaction", "confidence": 0.95}')

        # 2. Guardrail Prompt Response
        if "content safety filter" in prompt_str:
            user_part = prompt_str.split("user input:")[-1].split("unsafe criteria:")[0] if "user input:" in prompt_str else prompt_str
            if any(term in user_part for term in ["suicide", "kill myself", "harm myself", "lethal dose", "make a bomb", "cyanide"]):
                return "UNSAFE: Harmful or self-harm content detected. Please contact emergency services immediately (988 / 911)."
            return "SAFE"

        # 3. Vision Modality Classifier
        if "classify it as" in prompt_str:
            if "brain" in prompt_str or "mri" in prompt_str:
                return AIMessage(content='{"image_type": "BRAIN MRI", "reasoning": "Cranial axial slice MRI characteristics identified", "confidence": 0.95}')
            elif "skin" in prompt_str or "derm" in prompt_str:
                return AIMessage(content='{"image_type": "SKIN LESION", "reasoning": "Dermoscopic pigmented lesion characteristics identified", "confidence": 0.94}')
            else:
                return AIMessage(content='{"image_type": "CHEST X-RAY", "reasoning": "Pulmonary radiographic view identified", "confidence": 0.95}')

        # 4. Clinical SOAP Note Generator
        if "soap" in prompt_str or "subjective" in prompt_str:
            return AIMessage(content='''{
                "subjective": "Patient initiated clinical consultation regarding chest radiography, respiratory symptoms, or medical imaging triage.",
                "objective": "Consultation query analyzed by ClinicaGraph multi-agent diagnostic pipeline. Imaging or literature triage executed.",
                "assessment": "Clinical inquiry addressed with evidence-based guidance and automated radiographic workflow recommendations.",
                "plan": "1. Correlate with formal radiologist review. 2. Monitor pulmonary and systemic symptoms. 3. Seek prompt medical attention if acute respiratory distress occurs."
            }''')

        # 5. Question-Specific Clinical Responses
        if "chest x-ray" in prompt_str or "chest xray" in prompt_str:
            return AIMessage(content=(
                "### Diagnostic Modalities in Chest Radiography (CXR)\n\n"
                "Here are **5 primary methods and technologies** used in modern clinical Chest X-Ray detection and pathology assessment:\n\n"
                "1. **Deep Convolutional Neural Networks (DenseNet & ResNet)**:\n"
                "   - Automated classification of pulmonary opacities, consolidations, and infiltrates (e.g., COVID-19 vs. bacterial pneumonia vs. viral pneumonia).\n"
                "   - Utilizes transfer learning with pre-trained feature extractors optimized on datasets like CheXpert and NIH ChestX-ray14.\n\n"
                "2. **Class Activation Mapping (Grad-CAM & Saliency Overlays)**:\n"
                "   - Generates visual heatmap overlays identifying the exact anatomical lung zones (upper, middle, lower lobes) driving the diagnostic prediction.\n"
                "   - Critical for clinician interpretability and reducing algorithmic 'black-box' risks.\n\n"
                "3. **Dual-Energy Radiography (Subtracted Imaging)**:\n"
                "   - Rapid sequential acquisition of high- and low-energy X-rays to generate separate soft-tissue and bone-selective images, unmasking hidden nodules behind ribs.\n\n"
                "4. **Computer-Aided Detection (CAD) for Pulmonary Lesions**:\n"
                "   - Geometric pattern recognition and contour edge detection used to flag solitary pulmonary nodules (SPNs) and pneumothorax pleural line separation.\n\n"
                "5. **Multi-Agent Cross-Modal Evidence Retrieval (RAG)**:\n"
                "   - Real-time cross-referencing of radiographic findings with peer-reviewed clinical literature (e.g., RSNA/Fleischner Society guidelines) to provide differential diagnosis support.\n\n"
                "*Clinical Note: Automated radiographic AI findings require human-in-the-loop validation by a licensed radiologist.*"
            ))

        if prompt_str.strip() in ["hi", "hello", "hey", "greetings"]:
            return AIMessage(content=(
                "Hello! I am **ClinicaGraph AI**, your multi-agent clinical decision support system.\n\n"
                "How can I assist you today?\n"
                "- **Diagnostic Imaging**: Upload a Chest X-Ray, Brain MRI, or Skin Lesion for automated segmentation.\n"
                "- **Clinical Literature**: Ask about pathology protocols, COVID-19 pneumonia, or oncology research.\n"
                "- **Documentation**: Click **Clinical SOAP Note** above at any point to synthesize our session into an EHR-ready summary."
            ))

        # Default clinical consultation response
        return AIMessage(content=(
            "### ClinicaGraph Clinical Evaluation\n\n"
            "Thank you for your inquiry. In clinical practice, patient evaluation follows structured diagnostic pathways:\n\n"
            "- **Differential Assessment**: Reviewing symptom duration, severity, and physiological presentation.\n"
            "- **Diagnostic Imaging & Labs**: Confirming clinical suspicion with appropriate radiography (CXR, MRI, CT) or laboratory panels.\n"
            "- **Clinical Safety**: If experiencing persistent, worsening, or acute symptoms (e.g., severe dyspnea, chest pain, neurological deficits), please seek immediate in-person medical evaluation.\n\n"
            "*ClinicaGraph AI provides decision support and does not replace the professional judgement of a licensed clinician.*"
        ))


class MockEmbeddings(Embeddings):
    """Fallback simulated embedding generator generating deterministic 1536-dim vectors."""
    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        import hashlib
        import random
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)
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



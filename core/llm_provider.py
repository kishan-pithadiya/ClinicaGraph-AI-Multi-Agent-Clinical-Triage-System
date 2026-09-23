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
                if "brain" in query_part or "mri" in query_part:
                    return AIMessage(content='{"agent": "BRAIN_TUMOR_AGENT", "urgency": "ROUTINE", "reasoning": "Detected brain MRI image attachment", "confidence": 0.96}')
                elif "skin" in query_part or "lesion" in query_part or "rash" in query_part:
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

        # Extract user query if wrapped in prompt templates
        if "user:" in prompt_str:
            target_query = prompt_str.split("user:")[-1].split("clinical guidelines:")[0].strip()
        elif "query:" in prompt_str:
            target_query = prompt_str.split("query:")[-1].split("has image:")[0].strip()
        else:
            target_query = prompt_str

        # 5. Question-Specific Clinical Responses

        # A. Respiratory Pathogen Surveillance & Variants (2024-2026)
        if any(w in target_query for w in ["surveillance", "variant", "pathogen", "outbreak", "respiratory", "jn.1", "kp.2", "kp.3", "flirt"]):
            return AIMessage(content=(
                "### Global Respiratory Pathogen Surveillance & Variant Analysis (2024-2026)\n\n"
                "According to epidemiological data from the **WHO Global Influenza Surveillance and Response System (GISRS)** and **CDC NWSS** genomic surveillance, key respiratory pathogen characteristics include:\n\n"
                "1. **SARS-CoV-2 Lineage Evolution (JN.1 & KP.2 / KP.3 'FLiRT' Variants)**:\n"
                "   - **Dominance**: The JN.1 parent lineage evolved into sublineages designated KP.2, KP.3, and LB.1, collectively known as *FLiRT* variants based on spike mutations `F456L` and `R346T`.\n"
                "   - **Immune Evasion**: Exhibited significantly heightened antibody escape from prior monovalent XBB boosters, but conserved T-cell epitopes maintain protection against severe hospitalization.\n"
                "   - **Clinical Phenotype**: High transmissibility, sore throat, fatigue, moderate pyrexia, and GI symptoms (nausea/diarrhea in ~18% of cases), with lower rates of anosmia compared to ancestral strains.\n\n"
                "2. **Influenza A (H3N2, H1N1pdm09) & Avian H5N1 Monitoring**:\n"
                "   - Continuous cross-species surveillance active for Highly Pathogenic Avian Influenza (**HPAI H5N1 clade 2.3.4.4b**).\n"
                "   - Standard seasonal influenza managed via quadrivalent and updated cell-culture-based vaccines.\n\n"
                "3. **Respiratory Syncytial Virus (RSV)**:\n"
                "   - Significant reduction in infant and geriatric hospitalizations following the rollout of maternal bivalent RSV prefusion F vaccines (Abrysvo) and long-acting monoclonal antibodies (Nirsevimab).\n\n"
                "4. **Clinical Diagnostic Strategy**:\n"
                "   - **Multiplex RT-PCR Panels**: First-line recommendation for hospitalized patients to simultaneously differentiate Influenza A/B, RSV, and SARS-CoV-2.\n"
                "   - **CXR / Low-Dose CT**: Indicated when oxygen saturation drops (<95% on room air) to rule out viral pneumonia consolidation vs. secondary bacterial superinfection.\n\n"
                "*Notice: Epidemiological surveillance models are ground-truthed with global wastewater telemetry and genomic sequencing repositories.*"
            ))

        # B. Chest Radiography (CXR) Techniques
        if "chest x-ray" in target_query or "chest xray" in target_query or "cxr" in target_query:
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

        # C. COVID-19 Pneumonia Criteria & Symptoms
        if any(w in target_query for w in ["covid", "pneumonia", "consolidation", "infiltrate"]):
            return AIMessage(content=(
                "### Clinical & Radiographic Criteria for COVID-19 Pneumonia\n\n"
                "Diagnostic evaluation of viral pulmonary involvement follows standardized radiological guidelines:\n\n"
                "1. **Hallmark Radiographic Findings (CXR & CT)**:\n"
                "   - **Ground-Glass Opacities (GGOs)**: Bilateral, multifocal, predominantly peripheral and lower-zone subpleural distribution.\n"
                "   - **Consolidation**: Develops in severe stages, often with air bronchograms as exudates fill alveoli.\n"
                "   - **Absence of**: Pleural effusions, cavitary lesions, and discrete mediastinal lymphadenopathy (their presence suggests bacterial superinfection or tuberculosis).\n\n"
                "2. **Key Clinical Symptoms**:\n"
                "   - Persistent dry or productive cough, progressive dyspnea (shortness of breath).\n"
                "   - SpO2 desaturation (resting SpO2 < 94% on room air indicates moderate-to-severe disease).\n"
                "   - 'Silent Hypoxemia': Marked hypoxemia without proportional respiratory distress.\n\n"
                "3. **Biomarker Profile**:\n"
                "   - Elevated inflammatory markers: C-Reactive Protein (CRP), Ferritin, and D-Dimer.\n"
                "   - Lymphopenia (reduced absolute lymphocyte count is an early negative prognostic indicator).\n\n"
                "*Recommendation: Correlate radiographic imaging with supplemental oxygen needs and arterial blood gas (ABG) analysis.*"
            ))

        # D. ABCD Rule for Dermoscopy & Skin Lesions
        if any(w in target_query for w in ["abcd", "skin lesion", "melanoma", "dermoscopy", "mole"]):
            return AIMessage(content=(
                "### The ABCD Rule in Clinical Dermoscopy & Melanoma Screening\n\n"
                "The **ABCD criteria** are the gold standard for clinical differentiation between benign melanocytic nevi and malignant melanoma:\n\n"
                "- **A — Asymmetry**:\n"
                "  - One half of the pigmented lesion does not mirror the other half along either vertical or horizontal anatomical axes.\n\n"
                "- **B — Border Irregularity**:\n"
                "  - The lesion edges are scalloped, notched, ragged, or poorly circumscribed with abrupt pigment cut-offs.\n\n"
                "- **C — Color Variegation**:\n"
                "  - Non-uniform shades across the lesion: mixed presence of dark brown, jet black, reddish-pink, bluish-gray, or depigmented white zones.\n\n"
                "- **D — Diameter & Evolution**:\n"
                "  - Diameter exceeding **6 mm** (~size of a pencil eraser). In modern triage, **E (Evolution)** is added to track rapid changes in size, shape, elevation, or bleeding.\n\n"
                "*ClinicaGraph Diagnostic Vision uses U-Net segmentation to automatically compute border contour eccentricity and pigment symmetry.*"
            ))

        # E. Brain MRI Segmentation & Neuro-Radiology
        if any(w in target_query for w in ["brain", "mri", "tumor", "glioma", "segmentation", "neuro"]):
            return AIMessage(content=(
                "### Neuro-Radiology Brain MRI Segmentation Protocol\n\n"
                "ClinicaGraph's neuro-radiological agent analyzes cranial axial MRI scans through an automated computer vision pipeline:\n\n"
                "1. **Bilateral Filtering & Denoising**:\n"
                "   - Smooths background brain tissue while preserving critical high-frequency edges around intracranial boundaries.\n\n"
                "2. **Adaptive Intensity Thresholding**:\n"
                "   - Identifies abnormal contrast hyper-intensities characteristic of parenchymal mass lesions and peritumoral vasogenic edema.\n\n"
                "3. **Morphological Contour Extraction**:\n"
                "   - Calculates convex hull, lesion perimeter, circularity, and bounding spatial coordinates.\n\n"
                "4. **Clinical Verification Overlay**:\n"
                "   - Renders a color-coded bounding contour overlay directly on the axial slice to assist radiologists with localization and volumetric measurement.\n\n"
                "*Notice: Upload any cranial MRI scan via the 1-Click Samples or attachment button to generate live contour masks.*"
            ))

        # F. Greetings
        if target_query.strip() in ["hi", "hello", "hey", "greetings"]:
            return AIMessage(content=(
                "Hello! I am **ClinicaGraph AI**, your multi-agent clinical decision support system.\n\n"
                "How can I assist you today?\n"
                "- **Diagnostic Imaging**: Upload a Chest X-Ray, Brain MRI, or Skin Lesion for automated segmentation.\n"
                "- **Clinical Literature**: Ask about pathology protocols, COVID-19 pneumonia, or oncology research.\n"
                "- **Documentation**: Click **Clinical SOAP Note** above at any point to synthesize our session into an EHR-ready summary."
            ))

        # G. Dynamic Clinical Response for any other question
        return AIMessage(content=(
            f"### ClinicaGraph Clinical Assessment\n\n"
            f"**Regarding your clinical inquiry on {target_query[:60]}:**\n\n"
            f"In clinical decision support, evaluation of this presentation follows structured diagnostic principles:\n\n"
            f"- **Pathophysiological Evaluation**: Correlate symptom onset, duration, and physiological risk factors.\n"
            f"- **Diagnostic Modalities**: Consider confirmatory laboratory panels or targeted diagnostic imaging (e.g., Radiography, Ultrasound, or MRI) where clinically indicated.\n"
            f"- **Evidence-Based Management**: Align patient care with current clinical practice guidelines and institutional protocols.\n"
            f"- **Safety Precaution**: If acute red flags are present (e.g., chest pain, respiratory distress, acute focal neurological deficits), proceed immediately to emergency clinical triage.\n\n"
            f"*Disclaimer: ClinicaGraph AI is an autonomous clinical decision support system. It does not replace the professional judgement of a licensed healthcare provider.*"
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



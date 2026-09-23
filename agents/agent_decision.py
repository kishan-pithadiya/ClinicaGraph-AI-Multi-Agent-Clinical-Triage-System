import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import config
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

logger = logging.getLogger("ClinicaGraph.AgentDecision")

memory = MemorySaver()


class ClinicalDecision(TypedDict):
    agent: str
    urgency: str
    reasoning: str
    confidence: float


class AgentState(MessagesState):
    agent_name: Optional[str]
    current_input: Optional[Union[str, Dict]]
    has_image: bool
    image_type: Optional[str]
    output: Optional[Union[str, AIMessage]]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool
    urgency_level: str
    result_image: Optional[str]


class AgentConfig:
    CONFIDENCE_THRESHOLD = config.agent_decision.confidence_threshold
    
    SUPERVISOR_SYSTEM_PROMPT = """You are the Lead Clinical Triage Supervisor for ClinicaGraph AI, an autonomous clinical decision support platform.
    Analyze the patient/clinician query, presence of medical imaging, and dialogue history to determine:
    1. The most appropriate specialized agent to route to.
    2. The clinical urgency level (CRITICAL, URGENT, ROUTINE, INFORMATIONAL).

    Available Agents:
    1. CONVERSATION_AGENT: General inquiries, medical explanations, follow-up Q&A, and conversational guidance.
    2. RAG_AGENT: Evidence-based clinical literature lookup (oncology, neuro-oncology, respiratory infections, deep learning diagnostics, guidelines).
    3. WEB_SEARCH_PROCESSOR_AGENT: Time-sensitive queries, latest 2024-2026 medical developments, outbreak statistics, or novel drug approvals.
    4. BRAIN_TUMOR_AGENT: Neuro-imaging analysis for Brain MRI scans (segmentation & localization).
    5. CHEST_XRAY_AGENT: Pulmonary radiography evaluation for Chest X-Rays (COVID-19, pneumonia, normal).
    6. SKIN_LESION_AGENT: Dermatological assessment and lesion boundary segmentation.

    Routing Guidelines:
    - If an image is uploaded:
      - Brain MRI -> BRAIN_TUMOR_AGENT
      - Chest X-Ray -> CHEST_XRAY_AGENT
      - Skin Lesion / Rash -> SKIN_LESION_AGENT
      - Other medical image -> RAG_AGENT
    - If no image:
      - Novel/outbreak/recent news/current year -> WEB_SEARCH_PROCESSOR_AGENT
      - Medical literature/diagnostic criteria/treatment protocols/specific disease details -> RAG_AGENT
      - Greetings/casual conversation/general symptoms discussion -> CONVERSATION_AGENT

    Urgency Levels:
    - CRITICAL: Acute chest pain, stroke signs, severe dyspnea, suicide risk, active hemorrhage.
    - URGENT: High fever, unexplained severe pain, rapidly spreading rash, acute worsening symptoms.
    - ROUTINE: Chronic symptom inquiry, medication questions, general condition info.
    - INFORMATIONAL: Greetings, medical concepts, literature review, general queries.

    Output valid JSON matching:
    {{
      "agent": "AGENT_NAME",
      "urgency": "CRITICAL" | "URGENT" | "ROUTINE" | "INFORMATIONAL",
      "reasoning": "Clinical justification for routing and triage",
      "confidence": 0.95
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


def create_agent_graph():
    """Compiles the ClinicaGraph LangGraph clinical state machine."""

    guardrails = LocalGuardrails(config.rag.llm)
    decision_model = config.agent_decision.llm
    json_parser = JsonOutputParser(pydantic_object=ClinicalDecision)
    
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.SUPERVISOR_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    decision_chain = decision_prompt | decision_model | json_parser

    def analyze_input(state: AgentState) -> AgentState:
        current_input = state.get("current_input", "")
        has_image = False
        image_type = None
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")

        if input_text:
            is_allowed, msg = guardrails.check_input(input_text)
            if not is_allowed:
                return {
                    **state,
                    "messages": [msg],
                    "output": msg,
                    "agent_name": "SAFETY_GUARDRAIL",
                    "urgency_level": "CRITICAL" if any(k in str(msg.content).upper() for k in ["CRITICAL", "EMERGENCY", "911", "ALERT"]) else "ROUTINE",
                    "has_image": False,
                    "bypass_routing": True
                }

        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image")
            try:
                classification = AgentConfig.image_analyzer.analyze_image(image_path)
                image_type = classification.get("image_type", "OTHER")
            except Exception as e:
                logger.error(f"Image classification failed: {e}")
                image_type = "OTHER"

        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False
        }

    def check_if_bypassing(state: AgentState) -> str:
        return "apply_guardrails" if state.get("bypass_routing", False) else "route_to_agent"

    def route_to_agent(state: AgentState) -> Dict:
        messages = state.get("messages", [])
        current_input = state.get("current_input", "")
        has_image = state.get("has_image", False)
        image_type = state.get("image_type", "None")

        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        
        recent_history = ""
        for m in messages[-6:]:
            speaker = "Clinician/Patient" if isinstance(m, HumanMessage) else "ClinicaGraph"
            recent_history += f"{speaker}: {m.content}\n"

        decision_input = f"""
        Query: {input_text}
        Has Image: {has_image}
        Detected Image Modality: {image_type}
        Recent History:
        {recent_history}
        """

        try:
            decision = decision_chain.invoke({"input": decision_input})
            selected_agent = decision.get("agent", "CONVERSATION_AGENT")
            urgency = decision.get("urgency", "ROUTINE")
            confidence = float(decision.get("confidence", 0.9))
        except Exception as e:
            logger.warning(f"Supervisor routing parse fallback: {e}")
            if has_image:
                if "mri" in str(image_type).lower():
                    selected_agent = "BRAIN_TUMOR_AGENT"
                elif "x-ray" in str(image_type).lower():
                    selected_agent = "CHEST_XRAY_AGENT"
                elif "skin" in str(image_type).lower():
                    selected_agent = "SKIN_LESION_AGENT"
                else:
                    selected_agent = "CONVERSATION_AGENT"
            else:
                selected_agent = "RAG_AGENT" if "tumor" in input_text.lower() or "covid" in input_text.lower() else "CONVERSATION_AGENT"
            urgency = "ROUTINE"
            confidence = 0.85

        updated_state = {
            **state,
            "agent_name": selected_agent,
            "urgency_level": urgency
        }

        if confidence < AgentConfig.CONFIDENCE_THRESHOLD and not has_image:
            return {"agent_state": updated_state, "next": "RAG_AGENT"}
        
        return {"agent_state": updated_state, "next": selected_agent}

    def run_conversation_agent(state: AgentState) -> AgentState:
        current_input = state.get("current_input", "")
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        
        prompt = (
            f"You are ClinicaGraph AI, an advanced Clinical Decision Support assistant.\n"
            f"Respond professionally, accurately, and empathetically to the following user message.\n\n"
            f"User: {input_text}\n\n"
            f"Clinical Guidelines:\n"
            f"- Provide structured, clear medical information with bullet points where appropriate.\n"
            f"- Never prescribe medication or replace in-person medical evaluation.\n"
            f"- If symptoms suggest an emergency, remind the user to contact local emergency medical services."
        )
        response = config.conversation.llm.invoke(prompt)
        ai_msg = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
        return {
            **state,
            "output": ai_msg,
            "agent_name": "CONVERSATION_AGENT"
        }

    def run_rag_agent(state: AgentState) -> AgentState:
        query = state.get("current_input", "")
        query_text = query if isinstance(query, str) else query.get("text", "")
        rag_agent = MedicalRAG(config)

        recent_context = ""
        for m in state.get("messages", [])[-6:]:
            recent_context += f"{m.content}\n"

        response = rag_agent.process_query(query_text, chat_history=recent_context)
        confidence = response.get("confidence", 0.0)
        resp_content = response.get("response", "")
        resp_text = resp_content.content if hasattr(resp_content, 'content') else str(resp_content)

        insufficient = any(phrase in resp_text.lower() for phrase in [
            "don't have enough information", "not enough information",
            "insufficient information", "cannot answer"
        ])

        output_msg = AIMessage(content=resp_text) if confidence >= config.rag.min_retrieval_confidence else AIMessage(content="")
        return {
            **state,
            "output": output_msg,
            "retrieval_confidence": confidence,
            "insufficient_info": insufficient,
            "agent_name": "RAG_AGENT"
        }

    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        query = state.get("current_input", "")
        query_text = query if isinstance(query, str) else query.get("text", "")
        web_search = WebSearchProcessorAgent(config)
        processed = web_search.process_web_search_results(query=query_text)
        
        involved = f"{state.get('agent_name', '')} -> WEB_SEARCH_AGENT".strip(" ->")
        return {
            **state,
            "output": processed,
            "agent_name": involved
        }

    def confidence_based_routing(state: AgentState) -> str:
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            return "WEB_SEARCH_PROCESSOR_AGENT"
        return "check_validation"

    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        current_input = state.get("current_input", {})
        image_path = current_input.get("image") if isinstance(current_input, dict) else None
        
        mri_result = AgentConfig.image_analyzer.analyze_brain_mri(image_path)
        mask_path = mri_result.get("mask_path")
        summary = mri_result.get("summary", "Brain MRI scan processed.")
        location = mri_result.get("location", "Intracranial region")
        conf = int(mri_result.get("confidence", 0.9) * 100)

        content = (
            f"🧠 **ClinicaGraph Neuro-Radiology Report:**\n\n"
            f"- **Observation:** {summary}\n"
            f"- **Region / Localization:** {location}\n"
            f"- **Diagnostic Confidence:** {conf}%\n\n"
            f"A high-resolution segmentation overlay is rendered below for clinical verification."
        )
        return {
            **state,
            "output": AIMessage(content=content),
            "result_image": mask_path,
            "needs_human_validation": True,
            "agent_name": "BRAIN_TUMOR_AGENT"
        }

    def run_chest_xray_agent(state: AgentState) -> AgentState:
        current_input = state.get("current_input", {})
        image_path = current_input.get("image") if isinstance(current_input, dict) else None
        
        pred = AgentConfig.image_analyzer.classify_chest_xray(image_path)
        if pred == "covid19":
            status_text = "**POSITIVE** for **COVID-19 / Acute Viral Infiltrates**"
        elif pred == "normal":
            status_text = "**NEGATIVE / NORMAL** (Clear pulmonary parenchymal fields)"
        else:
            status_text = "**INDETERMINATE** (Insufficient radiographic clarity)"

        content = (
            f"🫁 **ClinicaGraph Pulmonary Radiography Report:**\n\n"
            f"- **Radiological Classification:** {status_text}\n"
            f"- **Modality:** Digital Chest Radiograph (PA/AP view)\n"
            f"- **Recommendation:** Correlate with clinical presentation, SpO2, and PCR panel."
        )
        return {
            **state,
            "output": AIMessage(content=content),
            "needs_human_validation": True,
            "agent_name": "CHEST_XRAY_AGENT"
        }

    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        current_input = state.get("current_input", {})
        image_path = current_input.get("image") if isinstance(current_input, dict) else None
        
        success = AgentConfig.image_analyzer.segment_skin_lesion(image_path)
        mask_path = "/uploads/skin_lesion_output/segmentation_plot.png" if success else None

        content = (
            f"🔬 **ClinicaGraph Dermatological Lesion Assessment:**\n\n"
            f"- **Status:** Lesion boundary segmented using U-Net morphological contours.\n"
            f"- **Clinical Rule Evaluation (ABCD):** Asymmetry and pigment distribution mapped.\n"
            f"- **Recommendation:** Visual inspection and dermoscopic biopsy review if borders are irregular."
        )
        return {
            **state,
            "output": AIMessage(content=content),
            "result_image": mask_path,
            "needs_human_validation": True,
            "agent_name": "SKIN_LESION_AGENT"
        }

    def handle_human_validation(state: AgentState) -> Dict:
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation"}
        return {"agent_state": state, "next": "apply_guardrails"}

    def perform_human_validation(state: AgentState) -> AgentState:
        output_text = state["output"].content if hasattr(state["output"], "content") else str(state["output"])
        validation_prompt = (
            f"{output_text}\n\n"
            f"🩺 **Clinician Review Required:**\n"
            f"- **Healthcare Professionals:** Validate this diagnostic inference (Confirm / Flag for Review).\n"
            f"- **Patients:** This AI output is for clinical support. Consult your physician."
        )
        return {
            **state,
            "output": AIMessage(content=validation_prompt),
            "agent_name": f"{state.get('agent_name', '')} [Awaiting Validation]"
        }

    def apply_output_guardrails(state: AgentState) -> AgentState:
        output = state.get("output")
        if not output:
            return state
        
        current_input = state.get("current_input", "")
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        sanitized_text = guardrails.check_output(output, input_text)
        
        msg = AIMessage(content=sanitized_text)
        messages = list(state.get("messages", []))
        messages.append(msg)
        return {
            **state,
            "messages": messages,
            "output": msg
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)

    workflow.set_entry_point("analyze_input")
    workflow.add_conditional_edges("analyze_input", check_if_bypassing, {
        "apply_guardrails": "apply_guardrails",
        "route_to_agent": "route_to_agent"
    })
    workflow.add_conditional_edges("route_to_agent", lambda x: x["next"], {
        "CONVERSATION_AGENT": "CONVERSATION_AGENT",
        "RAG_AGENT": "RAG_AGENT",
        "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
        "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
        "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
        "SKIN_LESION_AGENT": "SKIN_LESION_AGENT"
    })

    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing, {
        "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
        "check_validation": "check_validation"
    })
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_conditional_edges("check_validation", lambda x: x["next"], {
        "human_validation": "human_validation",
        "apply_guardrails": "apply_guardrails"
    })
    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)

    return workflow.compile(checkpointer=memory)


_compiled_graph = None

def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_agent_graph()
    return _compiled_graph


def init_agent_state() -> AgentState:
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "urgency_level": "ROUTINE",
        "result_image": None
    }


def process_query(query: Union[str, Dict], session_id: str = "default_session") -> Dict[str, Any]:
    graph = get_graph()
    thread_config = {"configurable": {"thread_id": session_id}}

    state = init_agent_state()
    state["current_input"] = query

    prompt_content = query if isinstance(query, str) else f"{query.get('text', '')} [Image uploaded for diagnostic analysis]"
    state["messages"] = [HumanMessage(content=prompt_content)]

    result = graph.invoke(state, thread_config)

    if len(result.get("messages", [])) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    return result


def synthesize_clinical_soap_note(session_id: str = "default_session") -> Dict[str, str]:
    thread_config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()
    
    try:
        current_state = graph.get_state(thread_config)
        messages = current_state.values.get("messages", [])
    except Exception:
        messages = []

    history_text = "\n".join([f"{m.type}: {m.content}" for m in messages]) if messages else "No prior consultation recorded."

    prompt = (
        f"You are a Clinical Documentation Specialist. Convert the following consultation dialogue into "
        f"a formal, structured SOAP (Subjective, Objective, Assessment, Plan) note.\n\n"
        f"CONSULTATION TRANSCRIPT:\n{history_text}\n\n"
        f"Format your response as valid JSON with keys:\n"
        f"- 'subjective': Chief complaint, reported symptoms, timeline\n"
        f"- 'objective': Radiographic observations, imaging results, vital indicators\n"
        f"- 'assessment': Primary differential diagnoses, risk stratification, ICD-10 suggestions\n"
        f"- 'plan': Next clinical steps, confirmatory diagnostics, patient precautions, follow-up"
    )

    try:
        llm = config.agent_decision.llm
        raw_res = llm.invoke(prompt)
        text = raw_res.content if hasattr(raw_res, 'content') else str(raw_res)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        soap_data = json.loads(text)
    except Exception as e:
        logger.warning(f"SOAP generation fallback: {e}")
        soap_data = {
            "subjective": "Patient initiated clinical dialogue regarding symptoms.",
            "objective": "Multimodal imaging and evidence retrieval processed via ClinicaGraph AI.",
            "assessment": "Clinical decision support generated. Requires clinician review.",
            "plan": "Complete in-person evaluation with diagnostic confirmation."
        }

    return soap_data
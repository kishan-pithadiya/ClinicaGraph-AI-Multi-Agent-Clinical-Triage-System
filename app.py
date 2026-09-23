"""
ClinicaGraph AI - FastAPI Application Server
Provides secure REST endpoints for multi-agent clinical decision support,
multimodal diagnostic image inference, human-in-the-loop validation,
audio transcription/synthesis, and automated SOAP note generation.
"""

import os
import uuid
import glob
import time
import threading
import logging
from io import BytesIO
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import requests
from werkzeug.utils import secure_filename

from config import config
from agents.agent_decision import process_query, synthesize_clinical_soap_note

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ClinicaGraph.Server")

# Initialize FastAPI app
app = FastAPI(
    title="ClinicaGraph AI: Multi-Agent Clinical Triage System",
    description="Autonomous Multimodal Clinical Decision Support and Multi-Agent Triage Platform",
    version="3.0.0"
)

# Storage directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
BRAIN_TUMOR_OUTPUT = "uploads/brain_tumor_output"
SPEECH_DIR = "uploads/speech"

for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, BRAIN_TUMOR_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static asset routes
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="templates")

# ElevenLabs client initialization (if key provided)
eleven_client = None
if config.speech.eleven_labs_api_key:
    try:
        from elevenlabs.client import ElevenLabs
        eleven_client = ElevenLabs(api_key=config.speech.eleven_labs_api_key)
    except Exception as e:
        logger.warning(f"ElevenLabs client init failed: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Background garbage collector for speech artifacts."""
    while True:
        try:
            for f in glob.glob(f"{SPEECH_DIR}/*.mp3") + glob.glob(f"{SPEECH_DIR}/*.webm"):
                if time.time() - os.path.getmtime(f) > 300:
                    os.remove(f)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        time.sleep(300)

threading.Thread(target=cleanup_old_audio, daemon=True).start()


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the ClinicaGraph clinical dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "config": config}
    )


@app.get("/health")
def health_check():
    """Service health probe."""
    return {
        "status": "healthy",
        "system": "ClinicaGraph AI",
        "version": "3.0.0",
        "triage_engine": "active"
    }


@app.get("/api/system-status")
def system_status():
    """Returns runtime diagnostic metrics for clinical telemetry."""
    return {
        "system": "ClinicaGraph AI Clinical Decision System",
        "version": "3.0.0",
        "active_llm": type(config.agent_decision.llm).__name__,
        "embedding_provider": type(config.rag.embedding_model).__name__,
        "vector_db": config.rag.vector_db_type,
        "reranker": config.rag.reranker_model,
        "available_agents": [
            "Clinical Triage Supervisor",
            "Conversational Agent",
            "Medical RAG Specialist",
            "Web Search Processor Agent",
            "Brain MRI Neuro-Radiology Agent",
            "Chest X-Ray Radiography Agent",
            "Dermatological Lesion Agent",
            "Human-in-the-Loop Validation"
        ]
    }


@app.post("/chat")
def chat(
    request: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None)
):
    """Process clinical text queries through the ClinicaGraph multi-agent state machine."""
    current_session = request.session_id or session_id or str(uuid.uuid4())
    response.set_cookie(key="session_id", value=current_session)

    try:
        response_data = process_query(request.query, session_id=current_session)
        response_text = ""
        if response_data.get("output"):
            out = response_data["output"]
            response_text = out.content if hasattr(out, 'content') else str(out)
        elif response_data.get('messages'):
            last_msg = response_data['messages'][-1]
            response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        result = {
            "status": "success",
            "session_id": current_session,
            "response": response_text,
            "agent": response_data.get("agent_name", "ClinicaGraph System"),
            "urgency": response_data.get("urgency_level", "ROUTINE"),
            "needs_validation": response_data.get("needs_human_validation", False)
        }


        # Check for generated image
        if response_data.get("result_image"):
            result["result_image"] = response_data["result_image"]

        return result
    except Exception as e:
        logger.error(f"Chat execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    """Process multimodal diagnostic image uploads (Brain MRI, Chest X-Ray, Dermoscopy)."""
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "Security Guardrail",
                "response": "Unsupported file format. Supported: PNG, JPG, JPEG, WEBP."
            }
        )

    file_bytes = await image.read()
    max_bytes = config.api.max_image_upload_size * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File size exceeds maximum threshold ({config.api.max_image_upload_size}MB)."
            }
        )

    current_session = session_id or str(uuid.uuid4())
    response.set_cookie(key="session_id", value=current_session)

    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    try:
        query_payload = {"text": text, "image": file_path}
        response_data = process_query(query_payload, session_id=current_session)
        last_msg = response_data['messages'][-1]
        response_text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        result = {
            "status": "success",
            "session_id": current_session,
            "response": response_text,
            "agent": response_data.get("agent_name", "Multimodal CV Agent"),
            "urgency": response_data.get("urgency_level", "ROUTINE"),
            "needs_validation": response_data.get("needs_human_validation", True)
        }

        # Check for visualization artifact
        if response_data.get("result_image"):
            result["result_image"] = response_data["result_image"]
        elif os.path.exists(os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")):
            result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"
        elif os.path.exists(os.path.join(BRAIN_TUMOR_OUTPUT, "mri_segmentation.png")):
            result["result_image"] = "/uploads/brain_tumor_output/mri_segmentation.png"

        # Cleanup input file
        try:
            os.remove(file_path)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.error(f"Image inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
def validate_output(
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Clinical Human-in-the-Loop verification endpoint."""
    current_session = session_id or str(uuid.uuid4())
    response.set_cookie(key="session_id", value=current_session)

    feedback_query = f"Clinician Human Validation: {validation_result}."
    if comments:
        feedback_query += f" Clinician Notes: {comments}"

    try:
        response_data = process_query(feedback_query, session_id=current_session)
        last_msg = response_data['messages'][-1]
        text = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        if validation_result.lower() in ['yes', 'confirm', 'approved']:
            status_label = "confirmed"
            header = "✅ **Inference Verified by Clinician:**"
        else:
            status_label = "flagged"
            header = "⚠️ **Inference Flagged for Secondary Clinical Review:**"

        return {
            "status": status_label,
            "header": header,
            "response": text,
            "comments": comments
        }
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/soap-note")
@app.post("/api/soap-note")
def generate_soap_note(
    request: Request,
    session_id: Optional[str] = Cookie(None)
):
    """Synthesizes structured clinical SOAP note from session transcript."""
    current_session = session_id or "default_session"
    try:
        soap_data = synthesize_clinical_soap_note(session_id=current_session)
        return {
            "status": "success",
            "session_id": current_session,
            "soap_note": soap_data
        }
    except Exception as e:
        logger.error(f"SOAP note synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribes verbal patient history using ElevenLabs Speech-to-Text."""
    if not eleven_client:
        return JSONResponse(status_code=503, content={"error": "Speech service unconfigured"})

    try:
        content = await audio.read()
        transcription = eleven_client.speech_to_text.convert(
            file=content,
            model_id="scribe_v1",
            language_code="eng"
        )
        return {"transcript": getattr(transcription, "text", "")}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Synthesizes clinical auditory feedback."""
    if not config.speech.eleven_labs_api_key:
        return JSONResponse(status_code=503, content={"error": "ElevenLabs API key not configured"})

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        res = requests.post(url, headers=headers, json={"text": request.text})
        if res.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "TTS synthesis failed"})

        out_path = os.path.join(SPEECH_DIR, f"{uuid.uuid4()}.mp3")
        with open(out_path, "wb") as f:
            f.write(res.content)

        return FileResponse(out_path, media_type="audio/mpeg", filename="clinical_audio.mp3")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("app:app", host=config.api.host, port=config.api.port, reload=config.api.debug)
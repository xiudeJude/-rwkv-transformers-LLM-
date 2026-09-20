from fastapi import APIRouter, HTTPException
from typing import Optional
from ..core.schemas import SingleStepInspectRequest, SingleStepInspectResponse, ModelMetadata
from ..core.config import settings
from ..models.transformer import TransformerEngine

router = APIRouter()

# Global inference engine instance
_engine: Optional[TransformerEngine] = None

def get_engine() -> TransformerEngine:
    global _engine
    if _engine is None:
        _engine = TransformerEngine()
        # Initialize default model
        _engine.load_model(settings.DEFAULT_TRANSFORMER, device=settings.DEVICE)
    return _engine

@router.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}

@router.get("/api/model/info", response_model=ModelMetadata)
async def model_info():
    try:
        engine = get_engine()
        return engine.get_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/inspect/single-step", response_model=SingleStepInspectResponse)
async def inspect_single_step(req: SingleStepInspectRequest):
    try:
        engine = get_engine()
        # If user explicitly requested a different model, load it if needed
        if req.model_name and req.model_name != engine.model_id:
            engine.load_model(req.model_name, device=settings.DEVICE)

        result = engine.inspect_single_step(
            prompt=req.prompt,
            layer=req.layer,
            head=req.head,
            top_k=req.top_k,
            temperature=req.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")

from ..core.schemas import CreateSessionRequest, StepRequest
from ..models.session import session_manager

@router.post("/api/session/create")
async def create_session(req: CreateSessionRequest):
    try:
        engine = get_engine()
        session = session_manager.create_session(engine)
        prefill_result = session.prefill(
            prompt=req.prompt,
            target_layer=req.layer,
            target_head=req.head
        )
        return {
            "status": "success",
            "session_id": session.session_id,
            "prefill_attention": prefill_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prefill failed: {str(e)}")

@router.post("/api/session/{session_id}/step")
async def session_step(session_id: str, req: StepRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        step_result = session.step(
            temperature=req.temperature,
            top_k=req.top_k,
            target_layer=req.layer,
            target_head=req.head
        )
        return {
            "status": "success",
            "session_id": session_id,
            **step_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

@router.post("/api/session/{session_id}/close")
async def close_session(session_id: str):
    success = session_manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "closed", "session_id": session_id}

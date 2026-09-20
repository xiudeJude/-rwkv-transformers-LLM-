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

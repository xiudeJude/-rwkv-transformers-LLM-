import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services import concept_lab

router = APIRouter(prefix="/api/concept", tags=["concept-lab"])


class MemoryGrowthRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Three architectures share this prompt")
    max_tokens: int = Field(12, ge=1, le=concept_lab.MAX_MEMORY_TOKENS)


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)


@router.post("/memory-growth")
async def memory_growth(req: MemoryGrowthRequest):
    try:
        return await asyncio.to_thread(
            concept_lab.run_memory_growth, req.prompt, req.max_tokens
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory growth experiment failed: {exc}")


@router.post("/tokenize")
async def tokenize(req: TokenizeRequest):
    try:
        return await asyncio.to_thread(concept_lab.tokenize_compare, req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tokenization comparison failed: {exc}")

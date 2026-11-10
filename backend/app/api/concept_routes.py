import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services import concept_lab

router = APIRouter(prefix="/api/concept", tags=["concept-lab"])


class MemoryGrowthRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Four architectures share this prompt")
    lengths: Optional[list[int]] = Field(
        None,
        description="prefill lengths; default 64/128/256/512, attention models also get 1024/2048",
    )


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)


class SampleRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    model_type: str = Field(
        ..., pattern="^(transformer|attention_baseline|rwkv|mamba)$"
    )


@router.post("/memory-growth")
async def memory_growth(req: MemoryGrowthRequest):
    try:
        from .routes import release_global_engines

        await asyncio.to_thread(release_global_engines)
        return await asyncio.to_thread(
            concept_lab.run_memory_growth, req.prompt, req.lengths
        )
    except concept_lab.ModelBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
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


@router.post("/sample-logits")
async def sample_logits(req: SampleRequest):
    try:
        from .routes import release_global_engines

        await asyncio.to_thread(release_global_engines)
        return await asyncio.to_thread(
            concept_lab.sample_topk_logits, req.prompt, req.model_type
        )
    except concept_lab.ModelBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sample logits failed: {exc}")

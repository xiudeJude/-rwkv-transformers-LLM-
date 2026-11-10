import os
import sys

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.api.concept_routes import router as concept_router
from app.core.config import settings
from app.services.model_gate import ModelBusyError

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="LLM Interpretability Workshop API (Transformer & RWKV)"
)

# Enable CORS for frontend local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ModelBusyError)
async def model_busy_handler(request: Request, exc: ModelBusyError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


app.include_router(router)
app.include_router(concept_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)

import asyncio
import orjson
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Response
from ..core.schemas import SingleStepInspectRequest, SingleStepInspectResponse, ModelMetadata
from ..core.config import settings
from ..models.transformer import TransformerEngine
from ..models.rwkv_engine import RWKVEngine
from ..models.rwkv_session import rwkv_session_manager

router = APIRouter()

# Global inference engine instances
_engine: Optional[TransformerEngine] = None
_rwkv_engine: Optional[RWKVEngine] = None

def get_engine() -> TransformerEngine:
    global _engine
    if _engine is None:
        _engine = TransformerEngine()
        # Initialize default model
        _engine.load_model(settings.DEFAULT_TRANSFORMER, device=settings.DEVICE)
    return _engine

def get_rwkv_engine() -> RWKVEngine:
    global _rwkv_engine
    if _rwkv_engine is None:
        _rwkv_engine = RWKVEngine()
        _rwkv_engine.load_model(settings.DEFAULT_RWKV, device=settings.DEVICE)
    return _rwkv_engine

@router.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}

@router.get("/api/model/info", response_model=ModelMetadata)
async def model_info(model_type: str = Query("transformer", description="'transformer' or 'rwkv'")):
    try:
        if model_type == "rwkv":
            engine = get_rwkv_engine()
        else:
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
            repetition_penalty=req.repetition_penalty,
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

@router.get("/api/session/{session_id}/attention")
async def get_session_attention(
    session_id: str,
    step: Optional[int] = Query(None, description="Step number up to which attention is retrieved"),
    layer: int = Query(3, description="Target full attention layer index"),
    head: int = Query(0, description="Target attention head index")
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        detail = session.get_attention_detail(step=step, layer=layer, head=head)
        return {
            "status": "success",
            **detail
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/session/{session_id}/stream")
async def session_websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": f"Session {session_id} not found or expired"})
        await websocket.close(code=1008)
        return

    stop_event = asyncio.Event()
    listener_task: Optional[asyncio.Task] = None

    try:
        # Wait for start command from client
        msg = await websocket.receive_json()
        action = msg.get("action")
        if action != "start":
            await websocket.send_json({"type": "error", "message": f"Expected action 'start', got '{action}'"})
            await websocket.close(code=1008)
            return

        # Extract and validate parameters
        temperature = float(msg.get("temperature", 0.55))
        top_k = int(msg.get("top_k", 10))
        repetition_penalty = float(msg.get("repetition_penalty", 1.2))
        max_new_tokens = int(msg.get("max_new_tokens", 50))
        target_layer = int(msg.get("layer", 3))
        target_head = int(msg.get("head", 0))

        if max_new_tokens <= 0 or max_new_tokens > 2048:
            await websocket.send_json({
                "type": "error",
                "message": f"非法参数: max_new_tokens 必须为正整数 (1 ~ 2048)，收到 {max_new_tokens}"
            })
            await websocket.close()
            return

        # Background listener for client stop/disconnect events
        async def client_listener():
            try:
                while not stop_event.is_set():
                    c_msg = await websocket.receive_json()
                    c_action = c_msg.get("action")
                    if c_action in ("close", "stop"):
                        stop_event.set()
                        break
            except WebSocketDisconnect:
                stop_event.set()
            except Exception:
                stop_event.set()

        listener_task = asyncio.create_task(client_listener())

        # Generation streaming loop
        tokens_generated = 0
        while tokens_generated < max_new_tokens and not session.is_finished:
            if stop_event.is_set():
                print(f"[WebSocket] Stop/disconnect signal detected for session {session_id}")
                break

            step_res = session.step(
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                target_layer=target_layer,
                target_head=target_head
            )
            tokens_generated += 1

            # Level 1 lightweight message with single-row attention slice
            level1_payload = {
                "type": "token_step",
                "step": step_res["step"],
                "token": step_res["token"],
                "token_id": step_res["token_id"],
                "token_prob": step_res["token_prob"],
                "topk_candidates": step_res["topk_candidates"],
                "attention_row": step_res["attention_slice"]["weights"],
                "is_finished": step_res["is_finished"]
            }
            await websocket.send_json(level1_payload)

            if step_res["is_finished"]:
                break

            await asyncio.sleep(0.001)

        # If hit max_new_tokens without EOS
        if not session.is_finished and not stop_event.is_set():
            session.is_finished = True
            await websocket.send_json({
                "type": "token_step",
                "step": session.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "attention_row": [],
                "is_finished": True,
                "finish_reason": "max_new_tokens"
            })


        # Wait while client decides next action or disconnects
        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for session {session_id}.")
    except Exception as e:
        print(f"[WebSocket Error] Exception in stream: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"Inference stream failed: {str(e)}"})
        except Exception:
            pass
    finally:
        stop_event.set()
        if listener_task and not listener_task.done():
            listener_task.cancel()
        print(f"[WebSocket] Releasing session {session_id} and freeing GPU VRAM.")
        session_manager.close_session(session_id)


# ==========================================
# RWKV-7 Session & Streaming Endpoints
# ==========================================

@router.post("/api/rwkv/session/create")
async def create_rwkv_session(req: CreateSessionRequest):
    try:
        engine = get_rwkv_engine()
        session = rwkv_session_manager.create_session(engine)
        prefill_result = session.prefill(
            prompt=req.prompt,
            target_layer=req.layer
        )
        return {
            "status": "success",
            "session_id": session.session_id,
            "prefill_state": prefill_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RWKV Prefill failed: {str(e)}")

@router.post("/api/rwkv/session/{session_id}/step")
async def session_rwkv_step(session_id: str, req: StepRequest):
    session = rwkv_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"RWKV Session {session_id} not found")
    try:
        step_result = session.step(
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            target_layer=req.layer
        )
        return {
            "status": "success",
            "session_id": session_id,
            **step_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RWKV Step failed: {str(e)}")

@router.post("/api/rwkv/session/{session_id}/close")
async def close_rwkv_session(session_id: str):
    success = rwkv_session_manager.close_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="RWKV Session not found")
    return {"status": "closed", "session_id": session_id}

@router.get("/api/session/{session_id}/rwkv_state")
@router.get("/api/rwkv/session/{session_id}/state")
async def get_session_rwkv_state(
    session_id: str,
    step: Optional[int] = Query(None, description="Step number (0 for prefill, 1..N for decode steps)"),
    layer: int = Query(0, description="Target layer index (0..23)"),
    head: Optional[int] = Query(None, description="Optional target head index (0..15). If omitted, returns all 16 heads.")
):
    session = rwkv_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"RWKV Session {session_id} not found")
    try:
        detail = session.get_rwkv_state(step=step, layer=layer, head=head)
        payload = {"status": "success", **detail}
        return Response(content=orjson.dumps(payload), media_type="application/json")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/rwkv/session/{session_id}/stream")
async def session_rwkv_websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = rwkv_session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": f"RWKV Session {session_id} not found or expired"})
        await websocket.close(code=1008)
        return

    stop_event = asyncio.Event()
    listener_task: Optional[asyncio.Task] = None

    try:
        msg = await websocket.receive_json()
        action = msg.get("action")
        if action != "start":
            await websocket.send_json({"type": "error", "message": f"Expected action 'start', got '{action}'"})
            await websocket.close(code=1008)
            return

        temperature = float(msg.get("temperature", 0.7))
        top_k = int(msg.get("top_k", 10))
        repetition_penalty = float(msg.get("repetition_penalty", 1.2))
        max_new_tokens = int(msg.get("max_new_tokens", 50))
        target_layer = int(msg.get("layer", 0))

        if max_new_tokens <= 0 or max_new_tokens > 2048:
            await websocket.send_json({
                "type": "error",
                "message": f"非法参数: max_new_tokens 必须为正整数 (1 ~ 2048)，收到 {max_new_tokens}"
            })
            await websocket.close()
            return

        async def client_listener():
            try:
                while not stop_event.is_set():
                    c_msg = await websocket.receive_json()
                    c_action = c_msg.get("action")
                    if c_action in ("close", "stop"):
                        stop_event.set()
                        break
            except WebSocketDisconnect:
                stop_event.set()
            except Exception:
                stop_event.set()

        listener_task = asyncio.create_task(client_listener())

        tokens_generated = 0
        while tokens_generated < max_new_tokens and not session.is_finished:
            if stop_event.is_set():
                print(f"[WebSocket RWKV] Stop/disconnect signal detected for session {session_id}")
                break

            step_res = session.step(
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                target_layer=target_layer
            )
            tokens_generated += 1

            level1_payload = {
                "type": "token_step",
                "step": step_res["step"],
                "token": step_res["token"],
                "token_id": step_res["token_id"],
                "token_prob": step_res["token_prob"],
                "topk_candidates": step_res["topk_candidates"],
                "state_summary": step_res["state_summary"],
                "delta": step_res["delta"],
                "layer": step_res["layer"],
                "is_finished": step_res["is_finished"]
            }
            await websocket.send_json(level1_payload)

            if step_res["is_finished"]:
                break

            await asyncio.sleep(0.001)

        if not session.is_finished and not stop_event.is_set():
            session.is_finished = True
            await websocket.send_json({
                "type": "token_step",
                "step": session.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "state_summary": [0.0] * 16,
                "delta": [0.0] * 16,
                "layer": target_layer,
                "is_finished": True,
                "finish_reason": "max_new_tokens"
            })

        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print(f"[WebSocket RWKV] Client disconnected for session {session_id}.")
    except Exception as e:
        print(f"[WebSocket RWKV Error] Exception in stream: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"RWKV Inference stream failed: {str(e)}"})
        except Exception:
            pass
    finally:
        stop_event.set()
        if listener_task and not listener_task.done():
            listener_task.cancel()
        print(f"[WebSocket RWKV] Releasing session {session_id} and freeing GPU VRAM.")
        rwkv_session_manager.close_session(session_id)



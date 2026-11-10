import asyncio
import orjson
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Response
from ..core.schemas import SingleStepInspectRequest, SingleStepInspectResponse, ModelMetadata, InternalState
from ..core.config import settings
from ..models.adapter_base import ModelAdapter
from ..models.adapter_registry import create_adapter
# Importing the adapter modules registers "transformer" / "rwkv" / "mamba"
# in the registry.
from ..models.transformer import TransformerAdapter
from ..models.rwkv_engine import RWKVAdapter
from ..models.mamba_engine import MambaAdapter, MAMBA_MAX_NEW_TOKENS
from ..models.rwkv_session import rwkv_session_manager

router = APIRouter()

# Global inference engine instances
_engine: Optional[ModelAdapter] = None
_rwkv_engine: Optional[ModelAdapter] = None
_mamba_engine: Optional[ModelAdapter] = None

def get_engine() -> ModelAdapter:
    global _engine
    if _engine is None:
        _engine = create_adapter(settings.MODEL_TYPE)
        # Initialize default model
        _engine.load_model(settings.DEFAULT_TRANSFORMER, device=settings.DEVICE)
    return _engine

def get_rwkv_engine() -> ModelAdapter:
    global _rwkv_engine
    if _rwkv_engine is None:
        _rwkv_engine = create_adapter(settings.RWKV_MODEL_TYPE)
        _rwkv_engine.load_model(settings.DEFAULT_RWKV, device=settings.DEVICE)
    return _rwkv_engine


def get_mamba_engine() -> ModelAdapter:
    global _mamba_engine
    if _mamba_engine is None:
        _mamba_engine = create_adapter(settings.MAMBA_MODEL_TYPE)
        _mamba_engine.load_model(settings.DEFAULT_MAMBA, device=settings.DEVICE)
    return _mamba_engine


def _dump_state(state: InternalState) -> Dict[str, Any]:
    """统一状态对象 -> JSON 友好的 dict。"""
    return state.model_dump()

@router.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}

@router.get("/api/model/info", response_model=ModelMetadata)
async def model_info(model_type: str = Query("transformer", description="'transformer', 'rwkv' or 'mamba'")):
    try:
        if model_type == "rwkv":
            engine = get_rwkv_engine()
        elif model_type == "mamba":
            engine = get_mamba_engine()
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
from ..models.session import session_manager, SessionManager

# Mamba uses the generic session wrapper; keep a separate manager so its
# TTL/LRU budget is independent from the Transformer sessions.
mamba_session_manager = SessionManager()

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
        state = session.export_state(step=0, layer=req.layer, head=req.head)
        return {
            "status": "success",
            "session_id": session.session_id,
            "prefill_attention": prefill_result,
            "state": _dump_state(state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prefill failed: {str(e)}")

@router.post("/api/session/{session_id}/step")
async def session_step(session_id: str, req: StepRequest):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired")
    try:
        step_result = session.step(
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            target_layer=req.layer,
            target_head=req.head
        )
        state = session.export_state(
            step=session.step_count,
            layer=req.layer,
            head=req.head,
            include_matrix=False,
        )
        return {
            "status": "success",
            "session_id": session_id,
            **step_result,
            "state": _dump_state(state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

@router.post("/api/session/{session_id}/close")
async def close_session(session_id: str):
    success = session_manager.close_session(session_id)
    # Idempotent by design: the WebSocket handler already releases the session on
    # disconnect, so a follow-up HTTP close from the frontend would otherwise 404.
    # Returning 200 keeps the client cleanup path silent and safe to retry.
    return {"status": "closed", "session_id": session_id, "already_closed": not success}

@router.get("/api/session/{session_id}/attention")
async def get_session_attention(
    session_id: str,
    step: Optional[int] = Query(None, description="Step number up to which attention is retrieved"),
    layer: int = Query(3, description="Target full attention layer index"),
    head: int = Query(0, description="Target attention head index")
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired")
    try:
        detail = session.get_attention_detail(step=step, layer=layer, head=head)
        state = session.export_state(step=step, layer=layer, head=head)
        return {
            "status": "success",
            **detail,
            "state": _dump_state(state)
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

            step_res = await asyncio.to_thread(
                session.step,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                target_layer=target_layer,
                target_head=target_head
            )
            if stop_event.is_set():
                print(f"[WebSocket] Stop signal processed after step for session {session_id}")
                break
            tokens_generated += 1

            # 统一状态导出与推理一样走线程，避免同步张量转换阻塞事件循环
            state = await asyncio.to_thread(
                session.export_state,
                step=step_res["step"],
                layer=target_layer,
                head=target_head,
                include_matrix=False,
            )

            # Level 1 lightweight message with single-row attention slice
            level1_payload = {
                "type": "token_step",
                "step": step_res["step"],
                "token": step_res["token"],
                "token_id": step_res["token_id"],
                "token_prob": step_res["token_prob"],
                "topk_candidates": step_res["topk_candidates"],
                "attention_row": step_res["attention_slice"]["weights"],
                "state": _dump_state(state),
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
        try:
            await websocket.close()
        except Exception:
            pass


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
        state = session.export_state(step=0, layer=req.layer, head=None, include_matrix=False)
        return {
            "status": "success",
            "session_id": session.session_id,
            "prefill_state": prefill_result,
            "state": _dump_state(state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RWKV Prefill failed: {str(e)}")

@router.post("/api/rwkv/session/{session_id}/step")
async def session_rwkv_step(session_id: str, req: StepRequest):
    session = rwkv_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"RWKV Session {session_id} not found or expired")
    try:
        step_result = session.step(
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            target_layer=req.layer
        )
        state = session.export_state(
            step=session.step_count,
            layer=req.layer,
            head=req.head,
            include_matrix=False,
        )
        return {
            "status": "success",
            "session_id": session_id,
            **step_result,
            "state": _dump_state(state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RWKV Step failed: {str(e)}")

@router.post("/api/rwkv/session/{session_id}/close")
async def close_rwkv_session(session_id: str):
    success = rwkv_session_manager.close_session(session_id)
    # Same idempotency contract as the Transformer close endpoint.
    return {"status": "closed", "session_id": session_id, "already_closed": not success}

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
        raise HTTPException(status_code=404, detail=f"RWKV Session {session_id} not found or expired")
    try:
        detail = session.get_rwkv_state(step=step, layer=layer, head=head)
        # 已有 legacy matrix；仅在指定 head（Level 2 弹窗）时让统一状态也携带矩阵，
        # 避免全层回填时重复传输 16x64x64 数据。
        state = session.export_state(
            step=step, layer=layer, head=head, include_matrix=head is not None
        )
        payload = {"status": "success", **detail, "state": _dump_state(state)}
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

            step_res = await asyncio.to_thread(
                session.step,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                target_layer=target_layer
            )
            if stop_event.is_set():
                print(f"[WebSocket RWKV] Stop signal processed after step for session {session_id}")
                break
            tokens_generated += 1

            state = await asyncio.to_thread(
                session.export_state,
                step=step_res["step"],
                layer=target_layer,
                head=None,
                include_matrix=False,
            )

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
                "state": _dump_state(state),
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
        try:
            await websocket.close()
        except Exception:
            pass



# ==========================================
# Mamba Session & Streaming Endpoints
# ==========================================

@router.post("/api/mamba/session/create")
async def create_mamba_session(req: CreateSessionRequest):
    try:
        engine = get_mamba_engine()
        session = mamba_session_manager.create_session(engine)
        prefill_result = session.prefill(
            prompt=req.prompt,
            target_layer=req.layer,
        )
        state = session.export_state(step=0, layer=req.layer, include_matrix=False)
        return {
            "status": "success",
            "session_id": session.session_id,
            "prefill_state": prefill_result,
            "state": _dump_state(state),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mamba Prefill failed: {str(e)}")


@router.post("/api/mamba/session/{session_id}/step")
async def session_mamba_step(session_id: str, req: StepRequest):
    session = mamba_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Mamba Session {session_id} not found or expired")
    try:
        step_result = session.step(
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            target_layer=req.layer,
        )
        state = session.export_state(
            step=session.step_count,
            layer=req.layer,
            include_matrix=False,
        )
        return {
            "status": "success",
            "session_id": session_id,
            **step_result,
            "state": _dump_state(state),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mamba Step failed: {str(e)}")


@router.post("/api/mamba/session/{session_id}/close")
async def close_mamba_session(session_id: str):
    success = mamba_session_manager.close_session(session_id)
    return {"status": "closed", "session_id": session_id, "already_closed": not success}


@router.get("/api/mamba/session/{session_id}/state")
async def get_mamba_session_state(
    session_id: str,
    step: Optional[int] = Query(None, description="Step number (0 for prefill, 1..N for decode steps)"),
    layer: int = Query(0, description="Target layer index"),
):
    session = mamba_session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Mamba Session {session_id} not found or expired")
    try:
        state = session.export_state(step=step, layer=layer, include_matrix=False)
        return {"status": "success", "session_id": session_id, "state": _dump_state(state)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/mamba/session/{session_id}/stream")
async def session_mamba_websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = mamba_session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": f"Mamba Session {session_id} not found or expired"})
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

        if max_new_tokens <= 0 or max_new_tokens > MAMBA_MAX_NEW_TOKENS:
            await websocket.send_json({
                "type": "error",
                "message": (
                    f"非法参数: Mamba 纯 PyTorch 参考路径的 max_new_tokens "
                    f"上限为 {MAMBA_MAX_NEW_TOKENS}，收到 {max_new_tokens}"
                )
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
                print(f"[WebSocket Mamba] Stop/disconnect signal detected for session {session_id}")
                break

            step_res = await asyncio.to_thread(
                session.step,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                target_layer=target_layer,
            )
            if stop_event.is_set():
                print(f"[WebSocket Mamba] Stop signal processed after step for session {session_id}")
                break
            tokens_generated += 1

            state = await asyncio.to_thread(
                session.export_state,
                step=step_res["step"],
                layer=target_layer,
                include_matrix=False,
            )
            payload = {
                "type": "token_step",
                "step": step_res["step"],
                "token": step_res["token"],
                "token_id": step_res["token_id"],
                "token_prob": step_res["token_prob"],
                "topk_candidates": step_res["topk_candidates"],
                "state": _dump_state(state),
                "is_finished": step_res["is_finished"],
            }
            await websocket.send_json(payload)

            if step_res["is_finished"]:
                break

            await asyncio.sleep(0.001)

        if not session.is_finished and not stop_event.is_set():
            session.is_finished = True
            state = await asyncio.to_thread(
                session.export_state,
                step=session.step_count,
                layer=target_layer,
                include_matrix=False,
            )
            await websocket.send_json({
                "type": "token_step",
                "step": session.step_count,
                "token": "",
                "token_id": -1,
                "token_prob": 0.0,
                "topk_candidates": [],
                "state": _dump_state(state),
                "is_finished": True,
                "finish_reason": "max_new_tokens",
            })

        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print(f"[WebSocket Mamba] Client disconnected for session {session_id}.")
    except Exception as e:
        print(f"[WebSocket Mamba Error] Exception in stream: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"Mamba Inference stream failed: {str(e)}"})
        except Exception:
            pass
    finally:
        stop_event.set()
        if listener_task and not listener_task.done():
            listener_task.cancel()
        print(f"[WebSocket Mamba] Releasing session {session_id} and freeing GPU VRAM.")
        mamba_session_manager.close_session(session_id)
        try:
            await websocket.close()
        except Exception:
            pass

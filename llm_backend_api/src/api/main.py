import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

# Load environment variables using python-dotenv if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; continue if not available
    pass


# ---------------------------
# App initialization and CORS
# ---------------------------

app = FastAPI(
    title="LLM Chat Backend API",
    description=(
        "Backend API for handling chat interactions with LLMs, including file and photo uploads, "
        "chat history, and model status endpoints. Supports pluggable LLM providers via environment "
        "configuration. See /openapi.json and /docs for details."
    ),
    version="0.1.0",
    contact={"name": "Chat Backend", "email": "support@example.com"},
    license_info={"name": "MIT"},
    terms_of_service="https://example.com/terms",
    openapi_tags=[
        {"name": "Health", "description": "Health and diagnostics"},
        {"name": "Chat", "description": "Chat operations and history"},
        {"name": "Uploads", "description": "File and photo upload endpoints"},
        {"name": "Models", "description": "LLM model info and status"},
        {"name": "Docs", "description": "API usage and websocket info"},
    ],
)

# Allow cross-origin for development; adjust in production with ENV
allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Storage setup (local disk)
# ---------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "runtime"))
UPLOADS_DIR = os.path.join(RUNTIME_DIR, "uploads")
ATTACHMENTS_DIR = os.path.join(RUNTIME_DIR, "attachments")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

# Expose uploads statically for debugging or UI previews
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="static_uploads")

# ---------------------------
# Simple in-memory storage for chat sessions and messages
# In a real app, replace with a database (dependency: chat_database)
# ---------------------------

class AttachmentMeta(BaseModel):
    """Metadata for an uploaded attachment."""
    id: str = Field(..., description="Unique attachment ID")
    filename: str = Field(..., description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    url: Optional[str] = Field(None, description="Public URL to access the file, if enabled")

class Message(BaseModel):
    """Represents a chat message in a session."""
    id: str = Field(..., description="Message ID")
    role: Literal["user", "assistant", "system"] = Field(..., description="Sender role")
    content: str = Field(..., description="Text content of the message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp (UTC)")
    attachments: List[AttachmentMeta] = Field(default_factory=list, description="Optional attachments metadata")
    model: Optional[str] = Field(None, description="Model used to generate this message (for assistant messages)")

class ChatSession(BaseModel):
    """Represents a chat session containing a sequence of messages."""
    session_id: str = Field(..., description="Unique session identifier")
    title: Optional[str] = Field(None, description="Optional user-friendly title")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp (UTC)")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp (UTC)")
    messages: List[Message] = Field(default_factory=list, description="Ordered list of messages")

# Simple in-memory store
_SESSIONS: Dict[str, ChatSession] = {}

# ---------------------------
# LLM Provider Abstraction (simple stubbed integration)
# ---------------------------

class ChatRequest(BaseModel):
    """Request payload for chat interactions."""
    session_id: Optional[str] = Field(None, description="Existing session ID; if omitted a new session is created")
    message: str = Field(..., description="User message to send")
    model: Optional[str] = Field(None, description="Target model identifier (provider specific)")
    attachments: List[str] = Field(
        default_factory=list,
        description="List of attachment IDs previously uploaded via /uploads endpoints"
    )
    system_prompt: Optional[str] = Field(
        None, description="Optional system prompt to steer the assistant behavior"
    )

class ChatResponse(BaseModel):
    """Response payload for chat interactions."""
    session_id: str = Field(..., description="Session ID used")
    message: Message = Field(..., description="Assistant response message")

class ListSessionsResponse(BaseModel):
    sessions: List[ChatSession]

class ListMessagesResponse(BaseModel):
    session_id: str
    messages: List[Message]

class UploadResponse(BaseModel):
    id: str
    filename: str
    content_type: Optional[str]
    size_bytes: Optional[int]
    url: Optional[str]

class ModelInfo(BaseModel):
    name: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    status: Literal["available", "degraded", "unavailable"] = Field(..., description="Operational status")
    context_window: Optional[int] = Field(None, description="Approximate context window tokens")
    description: Optional[str] = Field(None, description="Human friendly description")

class ModelStatusResponse(BaseModel):
    provider: str
    healthy: bool
    models: List[ModelInfo]

# -------------------------------------
# LLM provider selection and fake calls
# -------------------------------------

def _select_provider() -> str:
    """
    Pick provider based on environment variables.
    Supported env vars:
      - LLM_PROVIDER: openai|anthropic|azure_openai|ollama
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
      - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
      - OLLAMA_HOST
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    return provider

def _generate_assistant_reply(provider: str, prompt: str, model: Optional[str]) -> str:
    """
    Simulate calling the provider. In production, integrate actual SDK calls.
    """
    # This stub keeps things offline-friendly in CI while providing deterministic output.
    prefix = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "azure_openai": "Azure OpenAI",
        "ollama": "Ollama",
    }.get(provider, "LLM")
    model_label = f" [{model}]" if model else ""
    # A simple echo-style response
    return f"{prefix}{model_label} says: I received your message: '{prompt}'. How can I help further?"

# -------------------------------------
# Helpers for session and message mgmt
# -------------------------------------

def _ensure_session(session_id: Optional[str]) -> ChatSession:
    if session_id and session_id in _SESSIONS:
        sess = _SESSIONS[session_id]
        sess.updated_at = datetime.utcnow()
        return sess
    # Create a new session
    sid = session_id or str(uuid.uuid4())
    sess = ChatSession(session_id=sid, title=None)
    _SESSIONS[sid] = sess
    return sess

def _get_attachment_path(attachment_id: str) -> Optional[str]:
    """
    Return path to an uploaded file given an attachment ID.
    """
    for root in [UPLOADS_DIR, ATTACHMENTS_DIR]:
        candidate = os.path.join(root, attachment_id)
        if os.path.isfile(candidate):
            return candidate
    return None

def _build_attachment_meta(file_id: str, original_filename: str, content_type: Optional[str], size_bytes: Optional[int]) -> AttachmentMeta:
    return AttachmentMeta(
        id=file_id,
        filename=original_filename,
        content_type=content_type,
        size_bytes=size_bytes,
        url=f"/static/uploads/{file_id}",
    )

# ---------------------------
# Routes
# ---------------------------

# PUBLIC_INTERFACE
@app.get("/", tags=["Health"], summary="Health Check")
def health_check() -> Dict[str, Any]:
    """Simple health endpoint that returns a static message."""
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.get(
    "/docs/usage",
    tags=["Docs"],
    summary="API Usage Notes",
    description="Provides usage notes for REST endpoints. No websocket support in this implementation.",
)
def usage_notes() -> Dict[str, str]:
    """
    Returns plain text usage notes for clients.
    """
    return {
        "notes": (
            "Use /chat for sending messages, /uploads/file and /uploads/photo for attachments, "
            "/models for model list and status. Chat history is stored in-memory per session_id. "
            "Persist to a database in production."
        )
    }

# Chat endpoints
# PUBLIC_INTERFACE
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a chat message",
    description="Send a user message to the LLM, optionally attaching prior uploaded files/photos and returning the assistant response.",
    responses={
        200: {"description": "Assistant response with session_id"},
        400: {"description": "Invalid request"},
        500: {"description": "LLM provider error"},
    },
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint to send a user message and receive an assistant reply.
    - Creates a new session if session_id not provided.
    - Supports referencing attachment IDs uploaded via /uploads endpoints.
    - Stores message history in-memory keyed by session_id.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    session = _ensure_session(request.session_id)

    # Convert attachment IDs to metadata if they exist (best-effort)
    attachment_metas: List[AttachmentMeta] = []
    for att_id in request.attachments:
        path = _get_attachment_path(att_id)
        if path and os.path.isfile(path):
            # We don't have original filename stored; encode it into id on upload
            # For this stub, we keep only id and guessed metadata
            size = os.path.getsize(path)
            attachment_metas.append(
                _build_attachment_meta(att_id, original_filename=att_id, content_type=None, size_bytes=size)
            )

    # Optional system prompt
    if request.system_prompt:
        system_msg = Message(
            id=str(uuid.uuid4()),
            role="system",
            content=request.system_prompt,
            attachments=[],
        )
        session.messages.append(system_msg)

    # Store user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        role="user",
        content=request.message.strip(),
        attachments=attachment_metas,
    )
    session.messages.append(user_msg)

    # Provider call (stubbed)
    provider = _select_provider()
    try:
        assistant_text = _generate_assistant_reply(provider, prompt=request.message, model=request.model)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    assistant_msg = Message(
        id=str(uuid.uuid4()),
        role="assistant",
        content=assistant_text,
        model=request.model or provider,
    )
    session.messages.append(assistant_msg)
    session.updated_at = datetime.utcnow()

    return ChatResponse(session_id=session.session_id, message=assistant_msg)

# PUBLIC_INTERFACE
@app.get(
    "/chat/sessions",
    response_model=ListSessionsResponse,
    tags=["Chat"],
    summary="List chat sessions",
    description="Returns all in-memory chat sessions.",
)
def list_sessions() -> ListSessionsResponse:
    """List all sessions currently in memory."""
    return ListSessionsResponse(sessions=list(_SESSIONS.values()))

# PUBLIC_INTERFACE
@app.get(
    "/chat/{session_id}/messages",
    response_model=ListMessagesResponse,
    tags=["Chat"],
    summary="Get messages for a session",
    description="Returns ordered messages for a given session_id.",
)
def get_messages(session_id: str) -> ListMessagesResponse:
    """Retrieve messages for a given session."""
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return ListMessagesResponse(session_id=session_id, messages=session.messages)

# Upload endpoints
# PUBLIC_INTERFACE
@app.post(
    "/uploads/file",
    response_model=UploadResponse,
    tags=["Uploads"],
    summary="Upload a general file",
    description="Upload any file to be used as context for chat. Returns an attachment ID to reference in /chat.",
)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a generic file upload and save it to the uploads directory.
    Returns a stable attachment ID that clients can send via ChatRequest.attachments.
    """
    file_id = f"{uuid.uuid4()}__{file.filename}"
    save_path = os.path.join(UPLOADS_DIR, file_id)
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    size = os.path.getsize(save_path)
    meta = _build_attachment_meta(file_id, file.filename, file.content_type, size)
    return UploadResponse(**meta.model_dump())

# PUBLIC_INTERFACE
@app.post(
    "/uploads/photo",
    response_model=UploadResponse,
    tags=["Uploads"],
    summary="Upload a photo",
    description="Upload an image/photo to be used as context for chat. Returns an attachment ID to reference in /chat.",
)
async def upload_photo(photo: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a photo upload and save it to the uploads directory.
    Clients can use the returned id in subsequent chat calls.
    """
    # Simple content type check
    if photo.content_type and not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image uploads are allowed")

    photo_id = f"{uuid.uuid4()}__{photo.filename}"
    save_path = os.path.join(UPLOADS_DIR, photo_id)
    try:
        content = await photo.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    size = os.path.getsize(save_path)
    meta = _build_attachment_meta(photo_id, photo.filename, photo.content_type, size)
    return UploadResponse(**meta.model_dump())

# PUBLIC_INTERFACE
@app.get(
    "/models",
    response_model=ModelStatusResponse,
    tags=["Models"],
    summary="List models and provider status",
    description="Returns provider health and a small catalog of example models.",
)
def list_models() -> ModelStatusResponse:
    """
    Return provider health and example available models.
    In production, this should query the provider's list and status endpoints/SDKs.
    """
    provider = _select_provider()

    healthy = True
    models: List[ModelInfo] = []
    # Populate a simple catalog per provider
    if provider == "openai":
        models = [
            ModelInfo(name="gpt-4o-mini", provider="openai", status="available", context_window=128000, description="Multimodal mini"),
            ModelInfo(name="gpt-4o", provider="openai", status="available", context_window=200000, description="Multimodal flagship"),
        ]
    elif provider == "azure_openai":
        models = [
            ModelInfo(name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"), provider="azure_openai", status="available", description="Azure OpenAI deployment"),
        ]
    elif provider == "anthropic":
        models = [
            ModelInfo(name="claude-3-5-sonnet", provider="anthropic", status="available", context_window=200000),
            ModelInfo(name="claude-3-haiku", provider="anthropic", status="available", context_window=200000),
        ]
    elif provider == "ollama":
        models = [
            ModelInfo(name="llama3.1", provider="ollama", status="available", description="Local model via Ollama"),
            ModelInfo(name="mistral", provider="ollama", status="available"),
        ]
    else:
        healthy = False

    return ModelStatusResponse(provider=provider, healthy=healthy, models=models)

# Convenience: accept multipart chat with file in one request (optional for frontends)
# PUBLIC_INTERFACE
@app.post(
    "/chat/multipart",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send chat with inline file upload (convenience)",
    description="Allows sending a message and a single uploaded file at once. The uploaded file is saved and automatically attached.",
)
async def chat_multipart(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> ChatResponse:
    """
    Handles mixed multipart request containing a message and one optional file upload, then calls /chat logic.
    """
    att_ids: List[str] = []
    if file is not None:
        file_id = f"{uuid.uuid4()}__{file.filename}"
        save_path = os.path.join(UPLOADS_DIR, file_id)
        try:
            content = await file.read()
            with open(save_path, "wb") as f:
                f.write(content)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
        att_ids.append(file_id)

    req = ChatRequest(
        session_id=session_id,
        message=message,
        model=model,
        system_prompt=system_prompt,
        attachments=att_ids,
    )
    return chat(req)


# ---------------------------
# Global exception handlers (optional polish)
# ---------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


# ---------------------------
# Run helper (for uvicorn)
# ---------------------------

# PUBLIC_INTERFACE
def get_app() -> FastAPI:
    """Return the FastAPI app instance (for ASGI servers)."""
    return app

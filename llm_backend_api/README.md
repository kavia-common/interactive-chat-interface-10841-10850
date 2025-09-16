# LLM Backend API

FastAPI backend for a chat interface that interacts with LLMs, supports file/photo uploads, and exposes model status.

## Run locally

Install dependencies:
- Python 3.11+
- pip install -r requirements.txt

Run:
- uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Open:
- Docs: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json

## Environment

See ENVIRONMENT.md and .env.example. The app works offline with stubbed LLM responses by default.

## Storage

- Uploads saved to: llm_backend_api/runtime/uploads
- Static serving for previews: /static/uploads/<id>

## Endpoints

- GET /             Health check
- GET /docs/usage   Usage notes
- POST /chat        Send message (JSON)
- POST /chat/multipart  Send message + file in one request (multipart)
- GET /chat/sessions    List sessions
- GET /chat/{session_id}/messages  Get messages
- POST /uploads/file    Upload any file
- POST /uploads/photo   Upload an image
- GET /models           Models list and provider status

## Chat Flow

1. Upload files/photos via /uploads endpoints, note returned `id`.
2. Send /chat with JSON:
```
{
  "session_id": null,
  "message": "Hello",
  "model": "gpt-4o-mini",
  "attachments": ["<id-from-upload>"],
  "system_prompt": "You are helpful."
}
```
3. Use returned `session_id` for subsequent messages to retain history.

## Notes

- Sessions and messages are stored in-memory for demo; replace with database for persistence in production.
- LLM calls are stubbed in `_generate_assistant_reply`. Integrate providers as needed.

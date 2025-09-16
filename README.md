# interactive-chat-interface-10841-10850

This workspace contains:
- llm_backend_api: FastAPI backend for chat, uploads, and LLM integration.

Start backend locally:
- cd llm_backend_api
- pip install -r requirements.txt
- uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
- http://localhost:8000/docs
- http://localhost:8000/openapi.json
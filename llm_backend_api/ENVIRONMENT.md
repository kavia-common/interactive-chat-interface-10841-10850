# llm_backend_api Environment Variables

These variables configure the backend server, CORS, uploads, and the active LLM provider.

Mandatory for production usage if integrating real LLMs; for local development the app works with stubbed responses.

Core
- CORS_ALLOW_ORIGINS: Comma-separated list of allowed origins for CORS. Default: "*"

LLM Provider Selection
- LLM_PROVIDER: Select the provider integration to use. One of: openai, anthropic, azure_openai, ollama. Default: openai.

OpenAI
- OPENAI_API_KEY: API key for OpenAI requests.

Anthropic
- ANTHROPIC_API_KEY: API key for Anthropic requests.

Azure OpenAI
- AZURE_OPENAI_API_KEY: Azure OpenAI API key.
- AZURE_OPENAI_ENDPOINT: Base endpoint for Azure OpenAI.
- AZURE_OPENAI_DEPLOYMENT: Deployment name for the selected model.

Ollama (local)
- OLLAMA_HOST: Host URL (e.g., http://localhost:11434)

App Runtime
- No additional variables required. Files are stored under: llm_backend_api/runtime/uploads

Notes
- This template currently uses a stubbed LLM call for CI/offline environments. To integrate real providers, replace `_generate_assistant_reply` with actual SDK calls using the above env vars.

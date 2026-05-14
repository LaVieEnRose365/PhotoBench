
# =============================================================================
# LLM Model Paths (for local vLLM deployment)
# =============================================================================
model_paths = {
    "Hammer2.1-7b": "PATH/TO/Hammer2.1-7b",
    "ToolACE-2-Llama-3.1-8B": "PATH/TO/ToolACE-2-Llama-3.1-8B",
    "Qwen3-8B": "PATH/TO/Qwen3-8B",
    "Qwen3-32B": "PATH/TO/Qwen3-32B",
}

# =============================================================================
# vLLM Server Configuration
# =============================================================================
VLLM_DEFAULT_HOST = "0.0.0.0"
VLLM_DEFAULT_PORT = 8000

def get_vllm_base_url(host: str = "localhost", port: int = VLLM_DEFAULT_PORT) -> str:
    """Get the OpenAI-compatible base URL for vLLM server."""
    return f"http://{host}:{port}/v1"

# Default vLLM server URL (for local deployment)
VLLM_BASE_URL = get_vllm_base_url()

# =============================================================================
# API Configurations
# =============================================================================
# OpenAI API (for remote models like gpt-4o)
OPENAI_CONFIG = {
    "base_url": "YOUR_API_URL_HERE",
    # api_key should be set via environment variable OPENAI_API_KEY or passed as argument
}

# vLLM local API (for local models)
VLLM_CONFIG = {
    "base_url": VLLM_BASE_URL,
    "api_key": "EMPTY",  # vLLM doesn't require API key by default
}

# =============================================================================
# Claude (Anthropic) API Configuration
# =============================================================================
# For native Claude API access (without proxy)
# Set ANTHROPIC_API_KEY environment variable or pass api_key argument
ANTHROPIC_CONFIG = {
    # api_key should be set via environment variable ANTHROPIC_API_KEY
    # Models: claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-haiku-20240307, etc.
}

# =============================================================================
# Gemini (Google) API Configuration
# =============================================================================
# For native Gemini API access (without proxy)
# Set GOOGLE_API_KEY environment variable or pass api_key argument
GOOGLE_CONFIG = {
    # api_key should be set via environment variable GOOGLE_API_KEY
    # Models: gemini-1.5-pro, gemini-1.5-flash, etc.
}

# =============================================================================
# Unified Proxy API (supports multiple models via single endpoint)
# =============================================================================
# For accessing Claude, Gemini, OpenAI models through a unified proxy
PROXY_CONFIG = {
    "base_url": "YOUR_API_URL_HERE",
    # api_key should be set via environment variable or passed as argument
    # Supported models: gpt-4o, claude-*, gemini-*, deepseek-*, etc.
}

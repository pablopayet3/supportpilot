"""Central configuration: loads secrets from .env and builds the LLM.

Secrets never live in source code. Copy .env.example to .env and fill it in.
Switch model provider with PROVIDER=google (Gemini) or PROVIDER=azure (Azure OpenAI).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# LangSmith tracing (read automatically by LangChain / LangGraph)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "SupportPilot")

PROVIDER = os.getenv("PROVIDER", "google").strip().lower()
AGENT_VERSION = os.getenv("AGENT_VERSION", "v2")

AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://openai-api-management-gw.azure-api.net"
    "/openaiprodtest-clone/deployments/gpt-5-mini/chat/completions"
    "?api-version=2024-12-01-preview",
)
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-5-mini")

if PROVIDER == "azure":
    MODEL_NAME = AZURE_DEPLOYMENT
else:
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")


def get_llm(temperature: float = 0.0):
    """Return the chat model used by every node in the graph."""
    if PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI

        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise RuntimeError("AZURE_OPENAI_API_KEY is missing. Add it to your .env file.")
        # gpt-5 family models only support the default temperature, so it is not passed here
        return AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_version="2024-12-01-preview",
            deployment_name=AZURE_DEPLOYMENT,
            max_retries=6,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        max_retries=6,  # free-tier Gemini keys are rate limited; retry politely
    )


def text_of(message) -> str:
    """Extract plain text from a model message (Gemini can return a list of parts)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)

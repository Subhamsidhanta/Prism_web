"""LLM Service for Prism: Local-first (Mistral 7B via llama.cpp) or Gemini API.

Modes:
    local: Uses Mistral 7B GGUF model via llama-cpp-python.
    api: Uses Gemini via HTTP.

Environment Variables:
    LLM_MODE=local|api
    LLM_PROVIDER=gemini
    LLM_API_KEY=...                   # API key for Gemini
    LLM_MODEL_NAME=gemini-2.5-flash   # Remote model name
    LLM_MODEL_PATH=models/llm/model.gguf  # Local model path (when LLM_MODE=local)

Public Interface (BaseLLM):
    is_ready() -> bool
    generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str
    answer_question(context: str, question: str) -> str
    model_name property
    n_ctx attribute

Downstream code imports `mistral_llm` (always the correct LLM for the current mode).
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class BaseLLM:
    n_ctx: int = 8192
    model_name: str = "unknown"
    _last_error: Optional[str] = None

    def is_ready(self) -> bool:
        raise NotImplementedError

    def generate_response(self, prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str:
        raise NotImplementedError

    def answer_question(self, context: str, question: str) -> str:
        prompt = (
            "Answer the question based on the context below. Provide a complete and comprehensive answer with all relevant details. Do not stop mid-sentence.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self.generate_response(prompt, max_tokens=400, temperature=0.2)

    @property
    def last_error(self) -> Optional[str]:
        return getattr(self, "_last_error", None)

class GeminiLLM(BaseLLM):
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.api_key = os.getenv("LLM_API_KEY")
        self.model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
        self.n_ctx = 8192
        self._last_error: Optional[str] = None
        self._model = None

        if self.provider != "gemini":
            self._last_error = "LLM_PROVIDER must be 'gemini' for GeminiLLM"
            logger.error(self._last_error)
            return

        if not self.api_key:
            self._last_error = "LLM_API_KEY not set for Gemini"
            logger.error(self._last_error)
            return

        if genai:
            try:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                self._last_error = f"Gemini init error: {e}"
                logger.error(self._last_error)
        else:
            self._last_error = "google-generativeai library not installed"
            logger.error(self._last_error)

    def is_ready(self) -> bool:
        if not self.api_key:
            self._last_error = "LLM_API_KEY not set"
            return False
        if not genai:
            self._last_error = "google-generativeai library not installed"
            return False
        if not self._model:
            self._last_error = "Gemini model not initialized"
            return False
        return True

    def generate_response(self, prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str:
        if not self.is_ready():
            return f"Error: Gemini API not ready ({self._last_error})"
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            if hasattr(resp, 'text') and resp.text:
                return resp.text.strip()
            if hasattr(resp, 'candidates'):
                parts = []
                for c in resp.candidates:
                    for p in getattr(c, 'content', {}).parts:
                        if getattr(p, 'text', None):
                            parts.append(p.text)
                if parts:
                    return "\n".join(parts).strip()
            return "Error: Empty Gemini response"
        except Exception as e:
            self._last_error = f"Gemini request failed: {e}"
            return f"Error: {self._last_error}"

    def answer_question(self, context: str, question: str) -> str:
        max_context_length = 6000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        return super().answer_question(context, question)

def _build_llm() -> BaseLLM:
    logger.info("Initializing Gemini LLM provider (API mode only for Render/cloud)")
    llm = GeminiLLM()
    if not llm.is_ready():
        logger.warning(f"LLM not ready: {getattr(llm, '_last_error', 'unknown error')}")
    return llm

mistral_llm = _build_llm()
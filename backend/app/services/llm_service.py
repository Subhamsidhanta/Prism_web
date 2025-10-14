"""LLM Service abstraction supporting local GGUF (llama.cpp) and remote API providers.

Modes:
    LOCAL (default): Uses llama.cpp to load a GGUF model from disk.
    API: Uses an external provider (initially OpenAI-style API) via HTTP.

Environment Variables:
    LLM_MODE=local|api                # Select implementation (default: local)
    LLM_PROVIDER=openai               # Provider key (future: anthropic, groq, etc.)
    LLM_API_KEY=sk-...                # API key for provider
    LLM_MODEL_NAME=gpt-4o-mini        # Remote model name
    LLM_MODEL_PATH=...                # Local model path override
    PRISM_LLM_MODEL_PATH=...          # Alternate local model path env var
    LLM_API_BASE=https://api.openai.com/v1  # Override base URL

Public Interface (BaseLLM):
    is_ready() -> bool
    generate_response(prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str
    answer_question(context: str, question: str) -> str
    model_name property
    n_ctx attribute (approximate for API models if unknown)

Downstream code imports `mistral_llm` (kept for backwards compatibility) which now may be either LocalGGUFLLM or APILLM.
"""
import os
from typing import List, Optional, Dict, Any
import logging
import threading
import json
import time

try:
        from llama_cpp import Llama  # Local only; may not be available in API mode if dependency removed later
except ImportError:
        Llama = None

try:
        import requests  # Used for API-based providers
except ImportError:
        requests = None

# Optional Google Generative AI (Gemini) support
try:
    import google.generativeai as genai  # Modern library
except ImportError:
    genai = None
try:
    from google import genai as google_genai  # New client style
except ImportError:
    google_genai = None

logger = logging.getLogger(__name__)

class BaseLLM:
    """Abstract base class for LLM implementations."""
    model_path: Optional[str] = None
    n_ctx: int = 2048
    n_threads: int = 8
    model_name: str = "unknown"
    _last_error: Optional[str] = None

    def is_ready(self) -> bool:
        raise NotImplementedError

    def generate_response(self, prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str:
        raise NotImplementedError

    def answer_question(self, context: str, question: str) -> str:
        prompt = (
            "Answer the question based on the context below. Provide a complete and comprehensive answer with all relevant details. Do not stop mid-sentence.\n\n"  # noqa: E501
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self.generate_response(prompt, max_tokens=400, temperature=0.2)

    @property
    def last_error(self) -> Optional[str]:
        return getattr(self, "_last_error", None)


class LocalGGUFLLM(BaseLLM):
    """Local GGUF model via llama.cpp (original implementation)."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure model loads only once"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str = None, n_ctx: int = 2048, n_threads: int = 8):
        """
        Initialize Mistral 7B model using llama.cpp with performance optimizations
        
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (reduced to 2048 for speed)
            n_threads: Number of threads for inference (increased for speed)
        """
        if hasattr(self, 'initialized'):
            return
            
        # Allow env var overrides first
        env_model_path = os.getenv("LLM_MODEL_PATH") or os.getenv("PRISM_LLM_MODEL_PATH")
        self.model_path = model_path or env_model_path or self._find_model_path()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None
        self.initialized = True
        
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(
                f"Model not found at {self.model_path}. Set LLM_MODEL_PATH or place a .gguf under models/llm/."
            )

    @property
    def model_name(self) -> str:
        """Best-effort model name from the file path"""
        try:
            if self.model_path:
                return os.path.splitext(os.path.basename(self.model_path))[0]
        except Exception:
            pass
        return "unknown-model"
    
    def _find_model_path(self) -> str:
        """Find a GGUF model file in common locations, preferring Gemma then Mistral."""
        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Candidate directories to scan for any .gguf
        candidate_dirs = [
            os.path.join(cwd, "models", "llm"),
            os.path.join(os.path.dirname(os.path.dirname(script_dir)), "models", "llm"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))), "models", "llm"),
            os.path.abspath(os.path.join(script_dir, "..", "..", "models", "llm")),
            os.path.abspath(os.path.join(script_dir, "..", "..", "..", "models", "llm")),
        ]

        # Unique-ify and keep only existing dirs
        seen = set()
        dirs = []
        for d in candidate_dirs:
            if d and d not in seen:
                seen.add(d)
                if os.path.isdir(d):
                    dirs.append(d)

        ggufs: List[str] = []
        for d in dirs:
            try:
                for name in os.listdir(d):
                    if name.lower().endswith(".gguf"):
                        ggufs.append(os.path.abspath(os.path.join(d, name)))
            except Exception:
                continue

        # Also check a couple of legacy/fallback spots
        for p in [
            os.path.abspath("models/llm"),
            os.path.abspath("../models/llm"),
            os.path.abspath("../../models/llm"),
        ]:
            if os.path.isdir(p):
                try:
                    for name in os.listdir(p):
                        if name.lower().endswith(".gguf"):
                            ggufs.append(os.path.abspath(os.path.join(p, name)))
                except Exception:
                    pass

        # De-duplicate
        ggufs = sorted(set(ggufs))

        if ggufs:
            def score(path: str) -> tuple:
                fname = os.path.basename(path).lower()
                # Prefer gemma, then mistral; prefer higher quant (Q* order roughly)
                family_rank = 0
                if "gemma" in fname:
                    family_rank = 2
                elif "mistral" in fname:
                    family_rank = 1
                else:
                    family_rank = 0

                # Prefer K_M over K_S lightly, and Q6 > Q5 > Q4 > Q3
                quant_rank = 0
                if "q6" in fname:
                    quant_rank = 4
                elif "q5" in fname:
                    quant_rank = 3
                elif "q4" in fname:
                    quant_rank = 2
                elif "q3" in fname:
                    quant_rank = 1
                kv_rank = 1 if "_k_m" in fname else 0
                return (-family_rank, -quant_rank, -kv_rank, fname)

            # Sort by our preference ascending by negatives
            ggufs.sort(key=score)

            # After sorting by score, pick the first
            chosen = ggufs[0]
            logger.info("Discovered GGUF models:")
            for path in ggufs:
                logger.info(f" - {path}")
            logger.info(f"Selected model: {chosen}")
            return chosen

        # Final fallback: point to a typical path under CWD
        default_path = os.path.join(cwd, "models", "llm", "model.gguf")
        logger.warning(
            "No GGUF models found. Place your model under models/llm/ or set LLM_MODEL_PATH. "
            f"Expected default location: {default_path}"
        )
        return default_path
    
    def _load_model(self):
        """Load the model with performance optimizations"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            logger.info(f"Model file exists: {os.path.exists(self.model_path)}")
            logger.info(f"Model file size: {os.path.getsize(self.model_path) / (1024*1024*1024):.2f} GB")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,           # Reduced context for speed
                n_threads=self.n_threads,   # More threads for speed
                n_batch=512,                # Batch processing
                n_gpu_layers=0,             # CPU only for stability
                use_mmap=True,              # Memory mapping for faster loading
                use_mlock=False,            # Don't lock all memory
                verbose=True,               # Enable verbose for debugging
                f16_kv=True,                # Use half precision for KV cache
                logits_all=False,           # Don't compute logits for all tokens
            )
            logger.info("✅ Model loaded successfully with performance optimizations")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.llm = None
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready"""
        return self.llm is not None
    
    def generate_response(self, prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str:
        """
        Generate response from the model with speed optimizations
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (increased for complete responses)
            temperature: Sampling temperature (lower for faster, more focused responses)
            
        Returns:
            Generated response text
        """
        if not self.is_ready():
            return "Error: Model not loaded. Please check if the model file exists."
        
        try:
            # Choose a simple template; prefer Gemma chat format if detected
            formatted_prompt = self._format_prompt(prompt)
            
            # Optimized generation parameters for balanced speed and completeness
            response = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,                 # Slightly higher for better completeness
                top_k=50,                   # Increase vocabulary for better responses
                repeat_penalty=1.15,        # Stronger penalty to avoid repetition
                stop=["[/INST]", "[INST]", "</s>"],  # Remove \n\n stop to allow complete responses
                echo=False,
                stream=False
            )
            
            return response["choices"][0]["text"].strip()
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def _format_prompt(self, prompt: str) -> str:
        """Return a model-friendly prompt format based on detected model name."""
        name = (self.model_name or "").lower()
        if "gemma" in name:
            # Gemma chat template (simplified). Many Gemma instruct models in llama.cpp
            # use <start_of_turn>/<end_of_turn> markers.
            return (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        # Default to LLaMA/Mistral-style instruction format
        return f"[INST] {prompt} [/INST]"
    
    def answer_question(self, context: str, question: str) -> str:
        max_context_length = 3000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        return super().answer_question(context, question)

class APILLM(BaseLLM):
    """API-based LLM implementation (OpenAI style)."""
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.api_key = os.getenv("LLM_API_KEY")
        # Choose a sensible default model per provider
        if self.provider == "google":
            # Update default to newest stable fast model
            self.model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
        else:
            self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        self.n_ctx = 8192  # Approximate default for many hosted models
        self.session = None
        if requests:
            self.session = requests.Session()
            if self.api_key:
                self.session.headers.update({
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                })
        self._last_error: Optional[str] = None

        # Configure Google Generative AI if selected
        if self.provider == "google" and self.api_key:
            try:
                if genai:
                    genai.configure(api_key=self.api_key)
                elif google_genai:
                    # New experimental client usage
                    self._google_client = google_genai.Client(api_key=self.api_key)
                else:
                    logger.warning("Google provider selected but google-generativeai library not installed.")
            except Exception as e:
                self._last_error = f"Gemini init error: {e}"
                logger.error(self._last_error)
        else:
            self._google_client = None

    def is_ready(self) -> bool:
        if not self.api_key:
            self._last_error = "LLM_API_KEY not set"
            return False
        if self.provider == "openai":
            if not requests:
                self._last_error = "requests library not available"
                return False
            return True
        if self.provider == "google":
            # Need one of the genai libraries
            if not (genai or google_genai):
                self._last_error = "google-generativeai library not installed"
                return False
            return True
        self._last_error = f"Unsupported provider {self.provider}"
        return False

    def _openai_chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            resp = self.session.post(url, data=json.dumps(payload), timeout=60)
            if resp.status_code >= 400:
                self._last_error = f"API error {resp.status_code}: {resp.text[:200]}"
                return f"Error: {self._last_error}"
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self._last_error = str(e)
            return f"Error: {self._last_error}"

    def generate_response(self, prompt: str, max_tokens: int = 256, temperature: float = 0.4) -> str:
        if not self.is_ready():
            return f"Error: API not ready ({self._last_error})"
        if self.provider == "openai":
            return self._openai_chat(prompt, max_tokens, temperature)
        if self.provider == "google":
            return self._google_chat(prompt, max_tokens, temperature)
        return f"Error: Unsupported provider '{self.provider}'"

    def _google_chat(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            # Prefer modern generativeai library
            if genai:
                model = genai.GenerativeModel(self.model_name)
                # Gemini models don't always accept max_tokens; safety config can be added later
                resp = model.generate_content(prompt)
                if hasattr(resp, 'text') and resp.text:
                    return resp.text.strip()
                # Fallback: join parts
                if hasattr(resp, 'candidates'):
                    parts = []
                    for c in resp.candidates:
                        for p in getattr(c, 'content', {}).parts:
                            if getattr(p, 'text', None):
                                parts.append(p.text)
                    if parts:
                        return "\n".join(parts).strip()
                return "Error: Empty Gemini response"
            elif self._google_client:
                # Experimental new client style (from google import genai)
                response = self._google_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = getattr(response, 'text', None)
                if text:
                    return text.strip()
                return "Error: Empty Gemini response"
            else:
                return "Error: Gemini library not available"
        except Exception as e:
            self._last_error = f"Gemini request failed: {e}"
            return f"Error: {self._last_error}"

    def answer_question(self, context: str, question: str) -> str:
        max_context_length = 6000  # Use more since remote models can handle larger contexts
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        return super().answer_question(context, question)


def _build_llm() -> BaseLLM:
    mode = os.getenv("LLM_MODE", "local").lower()
    if mode == "api":
        logger.info("Initializing API LLM provider")
        api_llm = APILLM()
        if not api_llm.is_ready():
            logger.warning(f"API LLM not ready: {getattr(api_llm, '_last_error', 'unknown error')}")
        return api_llm
    # Fallback to local
    logger.info("Initializing local GGUF LLM")
    return LocalGGUFLLM()

# Backwards-compatible global instance name
mistral_llm = _build_llm()
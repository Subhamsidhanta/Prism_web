# Models Directory

This directory contains the machine learning models used by Prism.

## LLM Models (`llm/`)

### Mistral 7B Instruct v0.2

The Prism Q&A feature uses the Mistral 7B Instruct v0.2 model in GGUF format.

**Required File:** `mistral-7b-instruct-v0.2.Q4_K_M.gguf`

#### Download Instructions

```bash
# Create the directory if it doesn't exist
mkdir -p models/llm

# Download the model (4.1GB)
curl -L -o models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

#### Alternative Model Sizes

| Model | Size | Quality | Speed | Filename |
|-------|------|---------|-------|----------|
| Q2_K | 2.3GB | Lower | Fastest | `mistral-7b-instruct-v0.2.Q2_K.gguf` |
| Q4_K_M | 4.1GB | **Recommended** | Balanced | `mistral-7b-instruct-v0.2.Q4_K_M.gguf` |
| Q5_K_M | 4.8GB | Higher | Slower | `mistral-7b-instruct-v0.2.Q5_K_M.gguf` |
| Q8_0 | 7.2GB | Highest | Slowest | `mistral-7b-instruct-v0.2.Q8_0.gguf` |

#### Model Source

- **Repository:** [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- **Original Model:** [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **License:** Apache 2.0

## Embedding Models (`embed/`)

*Placeholder for future embedding models*

## Important Notes

⚠️ **Model files are excluded from git** - They are too large for GitHub and are listed in `.gitignore`

📥 **Download Required** - You must download the model files manually before running Prism

🔒 **Local Only** - All models run locally on your device for privacy

For detailed setup instructions, see [SETUP_QA.md](../SETUP_QA.md) in the project root.
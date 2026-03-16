# Serving LLM through OpenAI-Compatible Server with vLLM

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API. By default, it starts the server at http://localhost:8000. You can specify the address with --host and --port arguments. The server currently hosts one model at a time (OPT-125M in the command below) and implements list models, create chat completion, and create completion endpoints. We are actively adding support for more endpoints.

## Start the Server

```bash
vllm serve facebook/opt-125m
```

### Custom Chat Template
By default, the server uses a predefined chat template stored in the tokenizer. You can override this template by using the --chat-template argument:

```bash
vllm serve facebook/opt-125m --chat-template ./examples/template_chatml.jinja
```

## Query the Server

This server can be queried in the same format as OpenAI API.

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Create Completion
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

### Create Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

## API Documentation

Access the interactive Swagger UI at:
```
http://localhost:8000/docs
```

Access the raw OpenAPI JSON spec at:
```
http://localhost:8000/openapi.json
```

## Additional Server Options

### Specify Host and Port
```bash
vllm serve facebook/opt-125m --host 0.0.0.0 --port 8080
```

### Enable Tensor Parallelism
```bash
vllm serve facebook/opt-125m --tensor-parallel-size 2
```

### Set GPU Memory Usage
```bash
vllm serve facebook/opt-125m --gpu-memory-utilization 0.8
```

### Use Quantization
```bash
vllm serve facebook/opt-125m --quantization awq
```

# Start for production
```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 8192
```

## Python Client Example

You can also use the OpenAI Python client to interact with the server:

```python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",  # Required but unused
)

# Create a completion
completion = client.completions.create(
    model="facebook/opt-125m",
    prompt="Hello, how are you?",
    max_tokens=50,
    temperature=0.8
)

print(completion.choices[0].text)

# Create a chat completion
chat_completion = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=50,
    temperature=0.8
)

print(chat_completion.choices[0].message.content)
```
## Direct Python Usage (Without Server)

If you want to use vLLM directly in Python without starting a server:

```python
from vllm import LLM, SamplingParams

def main():
    prompt = "Hello, how are you?"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate([prompt], sampling_params)

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
```


## Production Deployment

### Using Screen
```bash
# Start server in background
screen -S vllm-server
vllm serve facebook/opt-125m --host 0.0.0.0 --port 8000

# Detach: Ctrl+A, D
# Reattach: screen -r vllm-server
```

### Using nohup
```bash
nohup vllm serve facebook/opt-125m --host 0.0.0.0 --port 8000 > vllm.log 2>&1 &
```

### Environment Variables
```bash
# Set GPU cache space
export VLLM_CPU_KVCACHE_SPACE=4  # in GB

# Disable logging
export VLLM_LOGGING_LEVEL=WARNING

# Start server
vllm serve facebook/opt-125m
```

## Troubleshooting

### Common Issues

**Out of Memory Error:**
```bash
# Reduce GPU memory usage
vllm serve facebook/opt-125m --gpu-memory-utilization 0.5

# Use quantization
vllm serve facebook/opt-125m --quantization awq

# Use smaller model
vllm serve distilgpt2
```

**Port Already in Use:**
```bash
# Use different port
vllm serve facebook/opt-125m --port 8080
```

**CUDA Out of Memory:**
```bash
# Use CPU mode
vllm serve facebook/opt-125m --device cpu

# Reduce batch size
vllm serve facebook/opt-125m --max-num-batched-tokens 1024
```

### Getting Help
```bash
# See all available options
vllm serve --help

# Check vLLM version
vllm --version
```

## Monitoring

### Health Checks
```bash
# Check if server is running
curl http://localhost:8000/health

# Check available models
curl http://localhost:8000/v1/models

# View server stats
curl http://localhost:8000/stats
```

## Quick Reference Commands

```bash
# Basic start
vllm serve facebook/opt-125m

# Production start
vllm serve meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 8192

# Test completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "prompt": "Hello", "max_tokens": 10}'

# View docs
open http://localhost:8000/docs
```

This complete guide covers everything from basic setup to production deployment for serving LLMs with vLLM.

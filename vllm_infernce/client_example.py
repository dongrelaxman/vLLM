from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="acdv",  # Required but unused
)

# Create a completion
completion = client.completions.create(
    model="facebook/opt-125m",
    prompt="Hello, how are you?",
    max_tokens=50,
    temperature=0.8
)

print("Completion result:")
print(completion.choices[0].text)

# Create a chat completion (requires chat template)
try:
    chat_completion = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        max_tokens=50,
        temperature=0.8
    )
    print("Chat completion result:")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(f"Chat completion failed: {e}")
    print("Note: Chat completions require a chat template. Use 'vllm serve facebook/opt-125m --chat-template ./template.jinja' to enable chat completions.")
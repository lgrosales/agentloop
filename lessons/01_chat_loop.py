"""
Lesson 1: Bare-Bones Chat Loop

An interactive CLI chatbot that maintains conversation history.
Demonstrates: API basics, message format, conversation state.
"""

from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
messages = []

print("Chat with Claude (type 'quit' to exit)")
print("-" * 40)

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=messages,
    )

    assistant_text = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_text})

    print(f"\nClaude: {assistant_text}")

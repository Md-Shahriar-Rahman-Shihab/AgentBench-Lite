import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.environ.get("AGENT_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
    temperature=0,
)

print("Model:", MODEL)
print("Answer:", response.choices[0].message.content)
print("Usage:", response.usage)
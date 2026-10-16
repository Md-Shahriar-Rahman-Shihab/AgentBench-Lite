from agent.llm import LLMClient

llm = LLMClient()
msgs = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]

for i in range(2):
    r = llm.chat(msgs)
    print(f"call {i}: {r.text} | cached={r.cached} | latency={r.latency_s:.3f}s | "
          f"tokens in/out/reasoning={r.prompt_tokens}/{r.completion_tokens}/{r.reasoning_tokens}")

r = llm.chat(msgs, salt="repeat-2")
print(f"with salt: {r.text} | cached={r.cached}")
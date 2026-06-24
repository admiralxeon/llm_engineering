# LLM Engineering — Weeks 1 & 2 Cheat Sheet (Anthropic track)

A one-page-ish revision reference. Everything is Claude-first, with the OpenAI equivalents noted where it helps.

---

## 1. THE CORE TRANSLATION TABLE (memorize this)

| Concept | OpenAI | Anthropic |
|---|---|---|
| Import / client | `from openai import OpenAI` → `OpenAI()` | `import anthropic` → `anthropic.Anthropic()` |
| API key | `OPENAI_API_KEY` (`sk-proj-…`) | `ANTHROPIC_API_KEY` (`sk-ant-…`) |
| Make a call | `client.chat.completions.create(...)` | `client.messages.create(...)` |
| **System prompt** | a `{"role":"system"}` message **inside** the list | a **separate** `system=` parameter |
| `max_tokens` | optional | **required** on every call |
| Read the reply | `response.choices[0].message.content` | `response.content[0].text` |
| Streaming | `stream=True`, loop chunks `delta.content` | `with client.messages.stream(...) as s:` loop `s.text_stream` |
| JSON output | `response_format={"type":"json_object"}` | **prefill** assistant turn with `{` (no flag exists) |
| `base_url` | n/a | native: bare `https://api.anthropic.com` (no `/v1/`); OpenAI-compat layer: **with** `/v1/` |
| Reasoning effort | `reasoning_effort="low"` | `thinking={"type":"enabled","budget_tokens":N}` |
| Count tokens | `tiktoken` (exact) | `client.messages.count_tokens(...)` |
| Tool/function format | `{"type":"function","function":{…,"parameters":{…}}}` | `{"name","description","input_schema":{…}}` |

**The three you got wrong on the quiz — drill these:**
1. System prompt: OpenAI = in the list; Anthropic = separate `system=`.
2. The SDK is a **thin HTTP wrapper** — the model runs on the provider's servers, never in the package.
3. Force JSON from Claude with **assistant prefill** (`{`), not a flag.

---

## 2. MINIMAL CLAUDE CALL (the shape of everything)

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5",        # cheap; "claude-sonnet-4-6" for quality
    max_tokens=1000,                 # REQUIRED
    system="You are a helpful assistant",   # separate, not in messages
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

Streaming version:
```python
with client.messages.stream(model="claude-haiku-4-5", max_tokens=1000,
                            system=sys, messages=msgs) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

JSON via prefill:
```python
messages=[{"role":"user","content":"...give JSON..."},
          {"role":"assistant","content":"{"}]      # prefill forces JSON
result = "{" + response.content[0].text             # glue the { back on
data = json.loads(result)
```

Extended thinking (note: `max_tokens` > `budget_tokens`; text is NOT `content[0]`):
```python
response = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000,
    thinking={"type":"enabled","budget_tokens":1024},
    messages=[{"role":"user","content":prompt}])
text = "".join(b.text for b in response.content if b.type == "text")
```

---

## 3. WEEK 1 — CONCEPTS

**Day 1 — Summarization.** Scrape a page, hand it to the model with a system prompt (task/persona) and a user prompt (the data). The summarizer pattern maps onto endless business tasks.

**Day 2 — What an API call really is.** The endpoint is just a URL you POST JSON to. The `anthropic`/`openai` package is a **wrapper** around that POST — no model inside it. Other providers (Ollama, Gemini, Groq…) expose **OpenAI-compatible** endpoints, so the OpenAI client talks to them just by changing `base_url`:
```python
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # free, local
```

**Day 4 — Tokenization & memory.**
- Tokenizers differ per model. `tiktoken` is **OpenAI's** — only an approximation for Claude/llama. Exact counts: Claude → `count_tokens` API; llama → its own tokenizer.
- **LLMs are stateless.** Each call is independent. "Memory" = resending the **whole conversation** every turn. Cost grows with every turn because you re-send everything.

**Day 5 — Multi-step pipeline.** Chain calls: pick relevant links (JSON) → scrape them → write a brochure → stream it. First taste of an agentic pattern.

---

## 4. WEEK 2 — CONCEPTS

**Day 1 — Many models, abstraction layers.**
- Inference-time scaling = extended thinking (see §2).
- LangChain: `from langchain_anthropic import ChatAnthropic` → `ChatAnthropic(model=...).invoke(messages)`.
- LiteLLM: `completion(model="anthropic/claude-sonnet-4-6", messages=...)` — provider prefix + cost reporting.

**Day 2 / 3 — Gradio.**
- `gr.Interface` = simple I/O; `gr.ChatInterface` = chatbot; `gr.Blocks` = custom layout.
- Stream in Gradio by `yield`-ing the accumulating string.
- The **system prompt is your main lever** for persona/context; you can modify it per-message before sending.

**Day 4 — Tools (the big one).** Flow: call with `tools=`; if `stop_reason=="tool_use"`, run the tool(s), append the model's request AND your results, call again; loop until it stops.
```python
# tool schema (Anthropic): input_schema, no wrapper
price_function = {"name":"get_ticket_price","description":"...",
  "input_schema":{"type":"object",
    "properties":{"destination_city":{"type":"string","description":"..."}},
    "required":["destination_city"]}}

while response.stop_reason == "tool_use":
    results = handle_tool_calls(response)                  # see below
    messages.append({"role":"assistant","content":response.content})  # echo request
    messages.append({"role":"user","content":results})    # results as a USER turn
    response = client.messages.create(..., tools=tools)

def handle_tool_calls(response):
    out = []
    for block in response.content:
        if block.type == "tool_use":
            city = block.input.get("destination_city")     # input is ALREADY a dict
            out.append({"type":"tool_result","tool_use_id":block.id,
                        "content": get_ticket_price(city)})
    return out
```
Key Anthropic differences vs OpenAI: `input_schema` (not `parameters`), `stop_reason=="tool_use"`, args are a **dict already** (no `json.loads`), results go back as a **user** turn with a `tool_result` block. Multiple/chained tool calls work for free (they're just multiple blocks).

**Day 5 — Multimodal agent.** Claude is text + vision-input only — **no image or audio generation**. Production shape: Claude reasons + calls tools; a separate specialist model makes media. Free substitutes used: `gTTS` for audio; HuggingFace (optional) for images.

---

## 5. GOTCHAS YOU ACTUALLY HIT (highest-value revision)

**Stale kernel state — the #1 Jupyter trap.** Edited a cell but the fix "didn't take"? The kernel uses the **last-executed** version of a function/variable, *regardless of cell position*. The `[n]` brackets show run order, not layout. **Fix: Kernel → Restart Kernel and Run All Cells.** Make this your first move whenever output looks wrong but code looks right. (Hit this 3× — schema, SQL, parser.)

**Schema is a contract.** Claude can only send arguments your `input_schema` declares. If the schema and your Python function signature disagree (e.g. a `set_price` tool missing a `price` property), the model sends the wrong/missing args and you get `None` / NULL / runtime errors that look like the model's fault but aren't. **Keep schema ⇄ function signature in lockstep.**

**Local models need defensive JSON parsing.** Small models (llama3.2) wrap valid JSON in prose and ```` ```json ```` fences. Don't trust `json.loads(raw)`. Extract the first balanced `{…}` object (fence-first, then brace-match), and fall back gracefully:
```python
fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
candidate = fence.group(1) if fence else text
# then find the first balanced {...} and json.loads it inside try/except
```

**`JSONDecodeError: Expecting value: line 1 column 1 (char 0)`** = `json.loads` got non-JSON at the very first char (empty string, or prose/fence before the `{`).

**Gradio 6 breaking changes** (newer than most tutorials):
- `gr.ChatInterface` **dropped `type="messages"`** (messages is now the only format). Remove it. (`gr.Chatbot` *kept* it.)
- History `content` is now a **list of blocks**, not a string. Flatten before sending to Anthropic:
```python
def to_text(content):
    if isinstance(content, str): return content
    if isinstance(content, dict): return content.get("text","")
    if isinstance(content, list):
        return "".join(b.get("text","") for b in content
                       if isinstance(b, dict) and b.get("type")=="text")
    return str(content)

messages = [{"role": h["role"], "content": to_text(h["content"])} for h in history]
```

---

## 6. DEBUGGING REFLEXES (the meta-skill)

1. **What's different between the call that worked and the one that broke?** (Worked on turn 1, broke on turn 2 → it's about *history*, not your logic.)
2. **Isolate the layer.** Row created but price NULL → the tool fired and handler ran; fault is *inside the function*, not the caller.
3. **`print`/`repr` the actual values** before assuming the caller is wrong. `repr()` reveals hidden fences/whitespace; printing `block.input` reveals what Claude really sent.
4. **Restart & Run All** before deep-debugging a "fix that didn't work."
5. **Library newer than the tutorial?** Suspect an API change, not your code.

---

## 7. QUICK SETUP REFERENCE

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...

# deps used across weeks 1-2
pip install anthropic openai python-dotenv beautifulsoup4 requests \
            tiktoken gradio gtts langchain-anthropic litellm

# local models (free)
ollama serve            # in a separate terminal
ollama pull llama3.2
```

Models: `claude-haiku-4-5` (cheap/fast), `claude-sonnet-4-6` (quality), `claude-opus-4-8` (hardest tasks). Verify current names at docs.claude.com if a string ever errors.

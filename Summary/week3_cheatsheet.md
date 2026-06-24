# LLM Engineering — Week 3 Cheat Sheet (Open-Source / HuggingFace)

Week 3 pivots from *renting* intelligence (APIs) to *running* it yourself: open-weight models via HuggingFace `transformers`, on a free GPU. Nothing here needs an Anthropic key — which is handy when you're low on tokens. The one exception (Day 5's generation step) is where Claude can slot in, noted at the end.

---

## The big picture: two levels of the HuggingFace API

| Level | What it is | Use when |
|---|---|---|
| **`pipeline()`** (high-level) | One function that hides tokenizing, model loading, and decoding | You just want the result (sentiment, summary, image…) |
| **Tokenizer + Model** (low-level) | You drive `AutoTokenizer` and `AutoModel…` yourself | You need control: custom generation, quantization, chat templates, streaming |

Days 2 → 4 walk you *down* that ladder: pipelines first, then the tokenizer, then the model.

---

## Day 1 — Google Colab (the environment)

Open models need a **GPU**, and Colab gives you one free.

- **Runtimes:** CPU / **T4 GPU** (free tier, enough for this course) / better GPUs on paid tiers. Set via Runtime → Change runtime type.
- `!nvidia-smi` shows your GPU and its memory (the T4 has ~15 GB — the constraint that makes *quantization* matter, see Day 4).
- `!pip install …` installs into the session; runtimes reset, so you reinstall each session.
- **Secrets:** store your `HF_TOKEN` in Colab's Secrets panel (key icon), not in code. Needed to download gated models (e.g. Llama).
- Mount Drive for files: `from google.colab import drive; drive.mount('/content/drive')`.

---

## Day 2 — Pipelines (the high-level API)

`pipeline("<task>")` returns a ready-to-call object. It picks a sensible default model, or you name one.

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis", device="cuda")
clf("I am thrilled with this course")        # -> [{'label': 'POSITIVE', 'score': 0.99}]
```

**Common tasks worth knowing:**

| Task string | Does |
|---|---|
| `sentiment-analysis` | positive / negative |
| `summarization` | long text → short |
| `translation_en_to_fr` | translate |
| `ner` | named-entity recognition |
| `question-answering` | answer from a context passage |
| `zero-shot-classification` | classify into labels you supply at runtime |
| `text-generation` | continue a prompt (uses a causal LM) |
| `automatic-speech-recognition` | audio → text (Whisper) — reused in Day 5 |
| `image-to-text`, `text-to-image` | captioning / image generation (via `diffusers`) |

Key flags: `device="cuda"` (or `device=0`) to use the GPU; `model="org/name"` to choose a specific Hub model. **Mental model:** a pipeline = tokenizer + model + post-processing bundled together.

---

## Day 3 — Tokenizers

Every model has its **own** tokenizer; you must use the matching one.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
ids = tok.encode("Hello world")     # text -> token ids
tok.decode(ids)                     # ids -> text
tok.tokenize("Hello world")         # see the actual sub-word pieces
```

**Concepts:**
- **Sub-word tokenization:** words split into pieces; counts differ per model (this is why `tiktoken` from Week 1 only *approximates* non-OpenAI models).
- **Vocab & special tokens:** each tokenizer has a fixed vocabulary and special markers — `BOS` (begin), `EOS` (end), `PAD`. `tok.special_tokens_map` shows them.
- **Chat templates (the important one).** Instruct/chat models expect a *specific* prompt format (where the system/user/assistant turns go, which special tokens wrap them). The tokenizer carries that template:

```python
messages = [{"role": "user", "content": "Explain a hashmap"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

Using the wrong template = garbled output. `apply_chat_template` is how you format prompts correctly for *any* open instruct model. (Different families — Llama, Phi, Qwen, Gemma, Mistral — all have different templates, and the tokenizer handles the difference for you.)

---

## Day 4 — Models (the low-level `transformers` API)

This is "the brains": load the actual weights and run generation yourself.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization so an 8B model fits on a free T4 (~15 GB)
quant = BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    quantization_config=quant,
)

inputs = tok.apply_chat_template(messages, return_tensors="pt",
                                 add_generation_prompt=True).to("cuda")
outputs = model.generate(inputs, max_new_tokens=200)
print(tok.decode(outputs[0]))
```

**Concepts:**
- **The full loop:** tokenizer **encodes** text → `model.generate()` produces token ids → tokenizer **decodes** back to text. Pipelines just hide this.
- **Quantization** (the make-or-break trick): full-precision 8B weights ≈ 32 GB and won't fit on a free GPU. Loading in **4-bit** (`bitsandbytes`) shrinks it ~4× with minor quality loss, so it fits. This is *the* technique for running big open models on small hardware.
- **`device_map="auto"`** spreads the model across available GPU/CPU memory automatically.
- **`max_new_tokens`** = how many tokens to generate (the open-model equivalent of `max_tokens`).
- **Streaming:** wrap with `TextStreamer(tok)` and pass `streamer=…` to `generate()` for token-by-token output.
- **Model families to recognize:** Llama (Meta), Phi (Microsoft), Gemma (Google), Qwen (Alibaba), Mistral/Mixtral. Picking among them is a Week 4 topic.

---

## Day 5 — Capstone: Meeting Minutes creator

Chains two models into a real product: **audio in → minutes out.**

1. **Transcribe** the audio with Whisper (a speech-to-text model):
```python
import whisper                       # pip install openai-whisper (runs locally/Colab)
stt = whisper.load_model("base")
transcript = stt.transcribe("meeting.mp3")["text"]
# or HF: pipeline("automatic-speech-recognition", model="openai/whisper-small")
```
2. **Generate structured minutes** from the transcript with an LLM (an open model like Llama, *or* Claude):
```python
# --- With Claude (your Anthropic deliverable for Week 3) ---
import anthropic
client = anthropic.Anthropic()
sys = ("You write professional meeting minutes in markdown: a summary, key discussion "
       "points, decisions, and action items with owners.")
resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000,
                              system=sys,
                              messages=[{"role": "user", "content": transcript}])
minutes = resp.content[0].text
```

**The architecture lesson** (this is the takeaway): a real system composes **specialist models** — a speech model for transcription, a language model for writing. The same pattern as Week 2's multimodal agent: each model does the one thing it's best at, and you wire them together. Claude is a drop-in for the *generation* half; Whisper handles the audio half.

---

## Open vs. closed models — when to use which

| | Open (HF, run yourself) | Closed (Claude/GPT API) |
|---|---|---|
| Cost | Free compute (your GPU); no per-token fee | Pay per token |
| Privacy | Data never leaves your machine | Sent to provider |
| Capability | Good, improving; usually below frontier | Top-tier reasoning |
| Setup | You manage GPU, memory, quantization | Just an API call |
| Best for | High volume, privacy, offline, cost control | Hardest reasoning, fastest to build |

For *your* situation (low on API tokens): Week 3's open models are the cost-free path for experimentation, exactly like Ollama was in Weeks 1–2 — just now you understand what's happening *under* Ollama (it's running quantized open models through this same machinery).

---

## One-line summary per day

- **D1 Colab** — free GPU environment; runtimes, secrets, `!nvidia-smi`.
- **D2 Pipelines** — `pipeline("task")`: the easy, high-level API for any task.
- **D3 Tokenizers** — `AutoTokenizer`; encode/decode, special tokens, and `apply_chat_template` for correct prompting.
- **D4 Models** — `AutoModelForCausalLM` + **4-bit quantization** to run big open models on small GPUs; the encode→generate→decode loop.
- **D5 Minutes** — chain Whisper (audio→text) + an LLM (Claude or open) into a product.

# 🍎 Apple Intelligence Hub

A multi-agent AI assistant for Apple products — routing every customer query to the right specialist automatically.

Try it live: [huggingface.co/spaces/viviancao/apple-AI-hub](https://huggingface.co/spaces/viviancao/apple-AI-hub)
Also available on Telegram: **@flora_apple_ai_bot**

---

## What It Does

Customers ask questions in natural language. The system automatically routes each query to the right specialist agent and replies with grounded, accurate information.

| Agent | Handles |
|---|---|
| 🛍️ Product | Stock, specs, pricing, buying decisions |
| 📜 Policy | Returns, warranties, shipping, FAQ |
| 🛠️ Tech | Troubleshooting, repairs, device support |
| 💼 Salary | Age/DOB-based salary estimates |
| 💬 General | Greetings, broad questions, anything else |

---

## Architecture

```
User (Web or Telegram)
        ↓
  Gradio UI / Telegram Bot (@flora_apple_ai_bot)
        ↓
   n8n Workflow (vivian-auto.app.n8n.cloud)
        ↓
  HF Space Gradio API
        ↓
  LangGraph Router  →  Specialist Agent  →  OpenAI gpt-4o-mini
        ↓                    ↓
  ChromaDB RAG        Internal Documents
        ↓
     Response
        ↓
  User (Web or Telegram)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Agents | LangGraph 0.2.56 + LangChain 0.3.x |
| RAG | ChromaDB + `sentence-transformers/all-mpnet-base-v2` |
| Document Loader | LangChain UnstructuredFileLoader |
| UI | Gradio 6.13.0 |
| Hosting | Hugging Face Spaces |
| Automation | n8n Cloud |
| Telegram Bot | @flora_apple_ai_bot |
| Salary Model | scikit-learn LinearRegression (joblib) |

---

## Knowledge Base

Documents are auto-classified into collections on startup:

```
apple data/
├── product/   → product_collection  (specs, inventory, prices)
├── policy/    → policy_collection   (returns, warranty, delivery)
└── tech/      → tech_collection     (troubleshooting, repair)
```

Any file not matching the above keywords goes into `general_collection`.

---

## n8n Workflow — Telegram Integration

The Telegram bot (@flora_apple_ai_bot) is powered by a 5-node n8n workflow:

```
[Telegram Trigger] → [HTTP Submit] → [Wait 5s] → [HTTP Poll] → [Code] → [Telegram Send]
```

### Node 1 — Telegram Trigger
- Type: `Telegram Trigger`
- Updates: `message`
- Credential: Telegram Bot API token (@flora_apple_ai_bot)

### Node 2 — HTTP Submit
- Method: `POST`
- URL: `https://viviancao-apple-ai-hub.hf.space/gradio_api/call/chat`
- Body (JSON):
```json
{ "data": ["{{ $json.message.text }}", []] }
```
- Returns: `{ "event_id": "abc123..." }`

### Node 3 — HTTP Poll
- Method: `GET`
- URL: `https://viviancao-apple-ai-hub.hf.space/gradio_api/call/chat/{{ $json.event_id }}`
- Response Format: `Text`
- Put Output in Field: `data`
- Timeout: `60000` ms

### Node 3.5 — Wait
- Resume: After time interval
- Amount: `5` seconds

### Node 4 — Code (JavaScript)
```javascript
const raw = $input.first().json.data;

const lines = raw.split('\n');
const dataLine = lines.find(l => l.startsWith('data: '));
const parsed = JSON.parse(dataLine.replace('data: ', ''));

const messages = parsed[1];
const lastMsg = messages[messages.length - 1];

const rawText = Array.isArray(lastMsg.content)
  ? lastMsg.content[0].text
  : lastMsg.content.text;

let reply = rawText;
reply = reply.split('<br>').slice(1).join(' ');
reply = reply.split('<details')[0];
reply = reply.replace(/<[^>]+>/g, '').trim();

return [{ json: { reply } }];
```

### Node 5 — Telegram Send
- Operation: `Send Message`
- Chat ID: `{{ $('Telegram Trigger').item.json.message.chat.id }}`
- Text: `{{ $json.reply }}`

---

## Environment Variables (HF Space Secrets)

| Secret | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (platform.openai.com/api-keys) |

---

## Local Setup

```bash
git clone https://huggingface.co/spaces/viviancao/apple-AI-hub
cd apple-AI-hub
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python app.py
```

---

## Keep the Space Awake (Optional)

The free HF Spaces tier sleeps after ~30 min of inactivity. Add a separate n8n workflow with a **Schedule Trigger** every 20 minutes:

- Method: `POST`
- URL: `https://viviancao-apple-ai-hub.hf.space/gradio_api/call/chat`
- Body: `{ "data": ["ping", []] }`

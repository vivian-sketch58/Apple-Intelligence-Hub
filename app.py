import html
import os
import re
import sys
import textwrap
from typing import TypedDict

import chromadb
import gradio as gr
import joblib
import pandas as pd


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# -----------------------------
# USER CONFIG
# -----------------------------
_HERE             = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR   = os.path.join(_HERE, "apple data")
DRIVE_DB_PATH     = os.path.join(_HERE, "chromaDB")
EMBEDDING_MODEL   = "sentence-transformers/all-mpnet-base-v2"
GEMINI_MODEL      = "gemini-2.5-flash"
SALARY_MODEL_PATH = os.path.join(_HERE, "salary_model.joblib")
RUN_ROUTER_TESTS  = False                      # set True to run diagnostic routing tests at startup

# FIX 1: validate API key at startup instead of silently assigning None
_api_key = os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    sys.exit("❌ GOOGLE_API_KEY environment variable is not set.")
os.environ["GOOGLE_API_KEY"] = _api_key

# LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=_api_key, temperature=0.2)

# Embeddings + persistent Chroma client
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
persistent_client = chromadb.PersistentClient(path=DRIVE_DB_PATH)

print("✅ Environment ready.")
print("SOURCE_DATA_DIR   =", SOURCE_DATA_DIR)
print("DRIVE_DB_PATH     =", DRIVE_DB_PATH)
print("MODEL             =", GEMINI_MODEL)
print("SALARY_MODEL_PATH =", SALARY_MODEL_PATH)

salary_model_artifact = joblib.load(SALARY_MODEL_PATH)

salary_model = salary_model_artifact["model"] if isinstance(salary_model_artifact, dict) and "model" in salary_model_artifact else salary_model_artifact
salary_feature = salary_model_artifact.get("feature", "Age") if isinstance(salary_model_artifact, dict) else "Age"
salary_target = salary_model_artifact.get("target", "Salary") if isinstance(salary_model_artifact, dict) else "Salary"

print("✅ salary_model.joblib loaded successfully")
print("Feature =", salary_feature)
print("Target  =", salary_target)

# BUILD KNOWLEDGE BASE
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

def classify_collection(filename: str) -> str:
    fname = filename.lower()
    if any(k in fname for k in ["product", "catalog", "inventory", "price"]):
        return "product_collection"
    elif any(k in fname for k in ["policy", "return", "shipping", "warranty", "faq"]):
        return "policy_collection"
    elif any(k in fname for k in ["tech", "repair", "support", "troubleshoot"]):
        return "tech_collection"
    else:
        return "general_collection"

def reset_selected_collections():
    for col_name in ["product_collection", "policy_collection", "tech_collection", "general_collection"]:
        try:
            persistent_client.delete_collection(col_name)
            print(f"🗑️ Deleted old collection: {col_name}")
        except Exception:
            pass

def load_documents_from_folder(folder_path: str):
    loader = DirectoryLoader(
        folder_path,
        glob="**/*",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=False
    )
    return loader.load()

# FIX 10: skip expensive rebuild when collections already exist
def build_knowledge_base(force_rebuild: bool = False):
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"❌ Folder not found: {SOURCE_DATA_DIR}")
        return

    if not force_rebuild:
        existing = {c.name for c in persistent_client.list_collections()}
        needed = {"product_collection", "policy_collection", "tech_collection", "general_collection"}
        if existing & needed:
            print("✅ Existing collections found. Skipping rebuild (pass force_rebuild=True to reset).")
            return

    reset_selected_collections()

    docs = load_documents_from_folder(SOURCE_DATA_DIR)
    if not docs:
        print("⚠️ No documents found.")
        return

    bucketed_docs = {
        "product_collection": [],
        "policy_collection": [],
        "tech_collection": [],
        "general_collection": [],
    }

    for doc in docs:
        source = doc.metadata.get("source", "unknown.txt")
        filename = os.path.basename(source)
        collection_name = classify_collection(filename)

        chunks = splitter.split_documents([doc])
        for c in chunks:
            c.metadata["source_file"] = filename
        bucketed_docs[collection_name].extend(chunks)

    for collection_name, chunk_list in bucketed_docs.items():
        if not chunk_list:
            continue

        _ = Chroma.from_documents(
            documents=chunk_list,
            embedding=embeddings,
            client=persistent_client,
            collection_name=collection_name
        )
        print(f"✅ Stored {len(chunk_list)} chunks in {collection_name}")

    print("\n🎉 Knowledge base build complete.")

build_knowledge_base()

# INSPECT RETRIEVAL (debugging utility — not called by the app)
def inspect_collection(collection_name: str, query: str, k: int = 3):
    try:
        db = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        docs = db.similarity_search(query, k=k)
        print(f"\n🔎 Collection: {collection_name}")
        print(f"Query: {query}")
        print("-" * 70)
        for i, d in enumerate(docs, start=1):
            print(f"[{i}] Source: {d.metadata.get('source_file', 'unknown')}")
            print(d.page_content[:400])
            print("-" * 70)
    except Exception as e:
        print(f"⚠️ Could not inspect {collection_name}: {e}")

# BUILD TOOLS
def _get_rag_docs(collection_name: str, query: str, k: int = 3):
    db = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    return db.similarity_search(query, k=k)

def _format_rag_docs(docs):
    formatted = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", "unknown")
        formatted.append(f"[{i}] Source: {src}\n{d.page_content}")
    return "\n\n".join(formatted)

# FIX 5: removed the `len(docs) < 2` short-circuit that triggered Gemini for any single-doc result
def _rag_looks_partial(docs):
    if not docs:
        return True
    joined = " ".join((d.page_content or "") for d in docs).strip()
    if len(joined) < 500:
        return True
    weak_markers = [
        "no internal documents found",
        "not available",
        "unknown",
        "not specified",
        "not mentioned"
    ]
    lowered = joined.lower()
    if sum(marker in lowered for marker in weak_markers) >= 2:
        return True
    return False

@tool
def gemini_supplement_tool(query: str) -> str:
    """Use Gemini as a supplemental knowledge tool for broader context when the internal RAG answer is incomplete."""
    try:
        prompt = (
            "Answer the question with concise, practical external/general knowledge. "
            "Do not claim access to internal company documents.\n\n"
            f"Question: {query}"
        )
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"Gemini supplement error: {e}"

# FIX 2: pass description directly to @tool so the LLM sees the correct per-tool description
def make_retrieval_tool(collection_name: str, tool_name: str, description: str):
    @tool(tool_name, description=description)
    def retrieval_tool(query: str) -> str:
        try:
            docs = _get_rag_docs(collection_name, query, k=3)
            if not docs:
                return "No internal documents found."
            return _format_rag_docs(docs)
        except Exception as e:
            return f"Error retrieving documents: {e}"
    return retrieval_tool

# FIX 2 (continued): same fix for hybrid tools
def make_hybrid_tool(collection_name: str, tool_name: str, description: str):
    @tool(tool_name, description=description)
    def hybrid_tool(query: str) -> str:
        try:
            docs = _get_rag_docs(collection_name, query, k=3)
            rag_text = _format_rag_docs(docs) if docs else "No internal documents found."
            needs_gemini = _rag_looks_partial(docs)

            if needs_gemini:
                gemini_text = gemini_supplement_tool.run(query)
                return (
                    "INTERNAL_RAG_RESULTS\n"
                    f"{rag_text}\n\n"
                    "GEMINI_SUPPLEMENT_RESULTS\n"
                    f"{gemini_text}\n\n"
                    "INSTRUCTIONS\n"
                    "Combine the internal RAG results and the Gemini supplement in the final answer. "
                    "Prioritize internal company facts when they conflict with Gemini content. "
                    "Use Gemini only to fill gaps, add broader context, or general knowledge."
                )

            return (
                "INTERNAL_RAG_RESULTS\n"
                f"{rag_text}\n\n"
                "INSTRUCTIONS\n"
                "The internal RAG results appear sufficient. Answer primarily from the internal documents."
            )
        except Exception as e:
            return f"Hybrid retrieval error: {e}"
    return hybrid_tool

product_rag_tool = make_retrieval_tool(
    "product_collection",
    "product_rag_tool",
    "Search internal product documents, inventory notes, specs, and prices."
)
policy_rag_tool = make_retrieval_tool(
    "policy_collection",
    "policy_rag_tool",
    "Search internal return policy, shipping, warranty, and FAQ documents."
)
tech_rag_tool = make_retrieval_tool(
    "tech_collection",
    "tech_rag_tool",
    "Search internal troubleshooting, repair, and technical support documents."
)
general_rag_tool = make_retrieval_tool(
    "general_collection",
    "general_rag_tool",
    "Search general internal documents."
)

product_hybrid_search = make_hybrid_tool(
    "product_collection",
    "product_hybrid_search",
    "Search internal product documents first, then add Gemini context if the internal answer is partial."
)
policy_hybrid_search = make_hybrid_tool(
    "policy_collection",
    "policy_hybrid_search",
    "Search internal policy documents first, then add Gemini context if the internal answer is partial."
)
tech_hybrid_search = make_hybrid_tool(
    "tech_collection",
    "tech_hybrid_search",
    "Search internal technical documents first, then add Gemini context if the internal answer is partial."
)
general_hybrid_search = make_hybrid_tool(
    "general_collection",
    "general_hybrid_search",
    "Search internal general documents first, then add Gemini context if the internal answer is partial."
)

@tool
def predict_salary(query: str) -> str:
    """Predict salary from age or DOB using the uploaded trained joblib model. Input examples: '35' or '1995-08-10'."""
    try:
        dob_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        if dob_match:
            dob = pd.to_datetime(dob_match.group(1), errors="coerce")
            if pd.isna(dob):
                return "Could not parse DOB. Please use YYYY-MM-DD."
            # FIX 4: leap-year-safe age calculation
            today = pd.Timestamp.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        else:
            age_match = re.search(r"\b(\d{1,3})\b", query)
            if not age_match:
                return "Please provide an age or a DOB in YYYY-MM-DD format."
            age = int(age_match.group(1))

        pred = salary_model.predict([[age]])[0]
        return f"Estimated salary for age {age}: {float(pred):,.2f}. This is a model estimate based only on age."
    except Exception as e:
        return f"Salary prediction error: {e}"

print("✅ Tools ready.")

# SPECIALIST AGENTS
# FIX 9: use a plain dict instead of globals() for agent lifecycle management
_agents: dict = {}

def initialize_agents():
    _agents["product"] = create_react_agent(
        llm,
        tools=[product_hybrid_search, product_rag_tool, gemini_supplement_tool],
        messages_modifier=(
            "You are the Product Agent. "
            "Handle buying questions, product specs, inventory, and prices. "
            "Use product_hybrid_search as your default first tool. "
            "That tool already combines internal RAG with Gemini when the internal answer is partial. "
            "When hybrid results contain both INTERNAL_RAG_RESULTS and GEMINI_SUPPLEMENT_RESULTS, synthesize both into one answer. "
            "Prioritize internal company facts if there is any conflict. "
            "Use the separate RAG or Gemini tools only if you need an extra follow-up lookup. "
            "Be concise and practical."
        )
    )

    _agents["policy"] = create_react_agent(
        llm,
        tools=[policy_hybrid_search, policy_rag_tool, gemini_supplement_tool],
        messages_modifier=(
            "You are the Policy Agent. "
            "Handle return rules, warranty, shipping, and FAQ. "
            "Use policy_hybrid_search as your default first tool. "
            "That tool combines internal RAG with Gemini only when the internal answer looks partial. "
            "When both sources are present, merge them into one answer and clearly favor internal policy over Gemini content. "
            "If the policy is still missing internally, say so clearly."
        )
    )

    _agents["tech"] = create_react_agent(
        llm,
        tools=[tech_hybrid_search, tech_rag_tool, gemini_supplement_tool],
        messages_modifier=(
            "You are the Tech Agent. "
            "Handle troubleshooting and technical explanations. "
            "Use tech_hybrid_search as your default first tool. "
            "That tool combines internal technical documents with Gemini only when the internal evidence is partial. "
            "Blend both sources into one practical answer, prioritizing internal product-specific guidance. "
            "Give safe, step-by-step advice."
        )
    )

    _agents["salary"] = create_react_agent(
        llm,
        tools=[predict_salary],
        messages_modifier=(
            "You are the Salary Agent. "
            "Handle salary estimation questions using the predict_salary tool. "
            "If the user provides a DOB, convert it via the tool. "
            "If the user gives only age, use it directly. "
            "Be clear that this is a model estimate, not a guaranteed salary."
        )
    )

    _agents["general"] = create_react_agent(
        llm,
        tools=[general_hybrid_search, general_rag_tool, gemini_supplement_tool],
        messages_modifier=(
            "You are the General Agent. "
            "Handle general chat, broad questions, and off-topic requests. "
            "Use general_hybrid_search as your default first tool. "
            "If that tool returns both internal RAG and Gemini supplement sections, combine them into one useful answer. "
            "Favor internal content over Gemini content whenever there is a conflict. "
            "Be helpful and concise."
        )
    )

def ensure_agents_initialized():
    if not _agents:
        initialize_agents()

ensure_agents_initialized()
print("✅ Agents initialized.")

# ROUTER NODE
VALID_ROUTES = ["product_agent", "policy_agent", "tech_agent", "salary_agent", "general_agent"]

def router_node(state):
    query = state["query"]

    router_prompt = '''
You are a routing classifier for a multi-agent AI system.

Classify the user query into exactly one of these labels:
- product_agent = products, catalog, specs, prices, inventory, buying decisions
- policy_agent = return policy, warranty, refund, shipping, FAQ, store rules
- tech_agent = troubleshooting, repair, setup, bugs, device support, technical issues
- salary_agent = salary prediction, salary estimate, pay estimate, age-based salary, DOB-based salary
- general_agent = greetings, jokes, writing, broad knowledge, anything else

Return ONLY one label from:
product_agent
policy_agent
tech_agent
salary_agent
general_agent
'''

    try:
        response = llm.invoke([
            SystemMessage(content=router_prompt),
            HumanMessage(content=query)
        ])
        decision = response.content.strip().lower().replace('"', "").replace("'", "")
    except Exception:
        decision = "general_agent"

    if decision not in VALID_ROUTES:
        decision = "general_agent"

    return {
        "next_node": decision,
        "debug_log": f"🚦 Router selected: {decision}"
    }

# FIX 8: router tests behind a flag — no LLM calls at startup in production
if RUN_ROUTER_TESTS:
    tests = [
        "Do you have iPhone 16 stock?",
        "What is your return deadline?",
        "My iPhone is overheating. What should I do?",
        "Predict salary for age 35.",
        "My DOB is 1995-08-10. Estimate salary.",
        "Tell me a joke about AI."
    ]
    for t in tests:
        out = router_node({"query": t})
        print(f"{t} -> {out['next_node']}")

# BUILD LANGGRAPH
# FIX 11: add history to state so agents have multi-turn context
class AgentState(TypedDict):
    query: str
    history: list
    response: str
    next_node: str
    debug_log: str

def extract_final_text(agent_result) -> str:
    try:
        msgs = agent_result.get("messages", [])
        if msgs:
            last = msgs[-1]
            return getattr(last, "content", str(last))
    except Exception:
        pass
    return str(agent_result)

def _build_messages(state: dict) -> list:
    messages = []
    for turn in state.get("history", [])[-6:]:
        if isinstance(turn, dict) and turn.get("role") == "user":
            messages.append(HumanMessage(content=turn.get("content", "")))
    messages.append(HumanMessage(content=state["query"]))
    return messages

def product_agent_node(state):
    ensure_agents_initialized()
    result = _agents["product"].invoke({"messages": _build_messages(state)})
    return {
        "response": extract_final_text(result),
        "debug_log": state.get("debug_log", "") + "\n🛍️ Product Agent answered."
    }

def policy_agent_node(state):
    ensure_agents_initialized()
    result = _agents["policy"].invoke({"messages": _build_messages(state)})
    return {
        "response": extract_final_text(result),
        "debug_log": state.get("debug_log", "") + "\n📜 Policy Agent answered."
    }

def tech_agent_node(state):
    ensure_agents_initialized()
    result = _agents["tech"].invoke({"messages": _build_messages(state)})
    return {
        "response": extract_final_text(result),
        "debug_log": state.get("debug_log", "") + "\n🛠️ Tech Agent answered."
    }

def salary_agent_node(state):
    ensure_agents_initialized()
    result = _agents["salary"].invoke({"messages": _build_messages(state)})
    return {
        "response": extract_final_text(result),
        "debug_log": state.get("debug_log", "") + "\n💼 Salary Agent answered."
    }

def general_agent_node(state):
    ensure_agents_initialized()
    result = _agents["general"].invoke({"messages": _build_messages(state)})
    return {
        "response": extract_final_text(result),
        "debug_log": state.get("debug_log", "") + "\n💬 General Agent answered."
    }

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("product_agent", product_agent_node)
workflow.add_node("policy_agent", policy_agent_node)
workflow.add_node("tech_agent", tech_agent_node)
workflow.add_node("salary_agent", salary_agent_node)
workflow.add_node("general_agent", general_agent_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda state: state["next_node"])
workflow.add_edge("product_agent", END)
workflow.add_edge("policy_agent", END)
workflow.add_edge("tech_agent", END)
workflow.add_edge("salary_agent", END)
workflow.add_edge("general_agent", END)

app = workflow.compile()
print("✅ LangGraph app compiled.")

# GRADIO CHAT UI
AGENT_ICONS = {
    "product_agent":  ("🛍️", "Product",  "#f59e0b"),
    "policy_agent":   ("📜", "Policy",   "#6366f1"),
    "tech_agent":     ("🛠️", "Tech",     "#10b981"),
    "salary_agent":   ("💼", "Salary",   "#ec4899"),
    "general_agent":  ("💬", "General",  "#64748b"),
}

# FIX 11 (continued): pass history into the graph; FIX 6: escape log to prevent XSS
def chat_fn(message: str, history: list) -> str:
    try:
        result = app.invoke({
            "query": message,
            "history": history,
            "response": "",
            "next_node": "",
            "debug_log": ""
        })
        answer = result.get("response", "No answer returned.")
        log    = result.get("debug_log", "No debug log.")
        route  = result.get("next_node", "general_agent")
        icon, label, color = AGENT_ICONS.get(route, ("🤖", "Agent", "#64748b"))

        badge = (
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'background:{color}22;border:1px solid {color}55;'
            f'color:{color};border-radius:999px;padding:2px 10px;'
            f'font-size:0.75rem;font-weight:600;margin-bottom:10px;">'
            f'{icon} {label} Agent</span>'
        )
        trace = (
            f'<details style="margin-top:12px;opacity:0.6;font-size:0.78rem;">'
            f'<summary style="cursor:pointer;font-weight:600;">🧠 Agent Trace</summary>'
            f'<pre style="white-space:pre-wrap;margin-top:6px;">{html.escape(log)}</pre></details>'
        )
        return f"{badge}<br>{answer}{trace}"
    except Exception as e:
        return f"⚠️ Error: {html.escape(str(e))}"


CUSTOM_CSS = """
/* ── Google Font imports ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --surface2:  #1a1e29;
    --border:    #ffffff0f;
    --accent:    #7c6af7;
    --accent2:   #3ecfcf;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --radius:    16px;
    --glow:      0 0 40px #7c6af730;
}

/* ── Global reset ── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    margin: 0 !important;
}

/* ── Animated mesh background ── */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, #7c6af71a 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, #3ecfcf12 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 48px 24px 24px;
    position: relative;
    z-index: 1;
}

.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(1.8rem, 4vw, 2.8rem) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #fff 30%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0 0 10px !important;
}

.app-header p {
    color: var(--muted);
    font-size: 0.95rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Agent pills row ── */
.agent-pills {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    padding: 0 16px 28px;
}
.agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid;
    opacity: 0.75;
    transition: opacity .2s;
}
.agent-pill:hover { opacity: 1; }

/* ── Chatbot container ── */
#chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--glow), inset 0 1px 0 #ffffff08 !important;
}

/* ── Chat bubbles ── */
.message.user .bubble-wrap,
.message.user .message-bubble-border {
    background: linear-gradient(135deg, var(--accent), #5b52c4) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #fff !important;
    box-shadow: 0 4px 20px #7c6af740 !important;
}

.message.bot .bubble-wrap,
.message.bot .message-bubble-border {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: var(--text) !important;
}

/* ── Textbox ── */
#msg-box textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: border-color .2s, box-shadow .2s;
}
#msg-box textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px #7c6af720 !important;
    outline: none !important;
}
#msg-box textarea::placeholder { color: var(--muted) !important; }

/* ── Buttons ── */
#send-btn, #clear-btn {
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    transition: all .2s !important;
}
#send-btn {
    background: linear-gradient(135deg, var(--accent), #5b52c4) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 4px 16px #7c6af740 !important;
}
#send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px #7c6af760 !important;
}
#clear-btn {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}
#clear-btn:hover {
    border-color: #ffffff25 !important;
    color: var(--text) !important;
}

/* ── Example chips ── */
.gr-samples-table button, .sample-btn {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    transition: all .2s !important;
}
.gr-samples-table button:hover, .sample-btn:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
    background: #7c6af710 !important;
}

/* ── Accordion / footer ── */
.gr-accordion {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ffffff18; border-radius: 99px; }
"""

HEADER_HTML = """
<div class="app-header">
  <h1>⬡ Apple Intelligence Hub</h1>
  <p>A multi-agent AI assistant powered by LangGraph &amp; Gemini — routing every query to the right specialist automatically.</p>
</div>
<div class="agent-pills">
  <span class="agent-pill" style="color:#f59e0b;border-color:#f59e0b55;background:#f59e0b0d;">🛍️ Products</span>
  <span class="agent-pill" style="color:#6366f1;border-color:#6366f155;background:#6366f10d;">📜 Policy</span>
  <span class="agent-pill" style="color:#10b981;border-color:#10b98155;background:#10b9810d;">🛠️ Tech</span>
  <span class="agent-pill" style="color:#ec4899;border-color:#ec489955;background:#ec48990d;">💼 Salary</span>
  <span class="agent-pill" style="color:#64748b;border-color:#64748b55;background:#64748b0d;">💬 General</span>
</div>
"""

SAMPLE_QUESTIONS = [
    "Do you have iPhone 16 Pro in stock?",
    "What is your return policy?",
    "My MacBook is overheating — what should I do?",
    "Predict salary for someone born on 1992-05-15",
    "What's the difference between M3 and M4 chips?",
]

with gr.Blocks(title="Apple Intelligence Hub") as demo:
    gr.HTML(HEADER_HTML)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=480,
        show_label=False,
        avatar_images=(None, "https://em-content.zobj.net/source/apple/391/robot_1f916.png"),
        render_markdown=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            elem_id="msg-box",
            placeholder="Ask about products, policies, tech support, salary estimates…",
            show_label=False,
            scale=8,
            lines=1,
        )
        send = gr.Button("Send ↑", elem_id="send-btn", scale=1, min_width=90)
        clear = gr.Button("Clear", elem_id="clear-btn", scale=1, min_width=80)

    gr.Examples(
        examples=SAMPLE_QUESTIONS,
        inputs=msg,
        label="✨ Try an example",
    )

    def respond(message: str, chat_history: list):
        bot_reply = chat_fn(message, chat_history)
        chat_history = chat_history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": bot_reply},
        ]
        return "", chat_history

    send.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch(debug=True, css=CUSTOM_CSS)

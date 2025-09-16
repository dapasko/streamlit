"""
Streamlit Multi-Model Chat (Sonoma + DeepSeek) with:
- file upload (pdf/csv/json/txt/images/py)
- chunking/embedding & optional vector DB (local faiss or external Qdrant/Pinecone)
- summarization for long histories
- two selectable models: openrouter/sonoma-sky-alpha, deepseek/deepseek-chat-v3.1:free
- compatible with Streamlit Cloud (use st.secrets for OPENROUTER_API_KEY)
Notes:
- Optional features (vector DB) require extra packages from requirements_optional.txt
- Use secrets in Streamlit Cloud: OPENROUTER_API_KEY, (optional) PINECONE_KEY, PINECONE_ENV, QDRANT_URL, QDRANT_API_KEY
"""

import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import pandas as pd
import logging
import math
import os
import time

# Optional libs (vector store). We import lazily and handle missing packages gracefully.
_HAS_SENT_TRANSFORMERS = False
_HAS_FAISS = False
_HAS_QDRANT = False
_HAS_PINECONE = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENT_TRANSFORMERS = True
except Exception:
    _HAS_SENT_TRANSFORMERS = False

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    _HAS_QDRANT = True
except Exception:
    _HAS_QDRANT = False

try:
    import pinecone
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

# =========================
# Constants и Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB per file
CHUNK_TOKENS = 1500  # target chunk size in "approx tokens" (used for chunking by characters)
CHUNK_OVERLAP = 200
MAX_HISTORY = 40  # keep fairly large; summarization will compress older parts
SUPPORTED_EXT = ['txt', 'pdf', 'jpg', 'jpeg', 'png', 'csv', 'json', 'py', 'md']

# =========================
# Helper utilities
# =========================
def chunk_text(text: str, chunk_size=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """Разбить текст на чанки по символам приближенно, учитывая overlap."""
    if not text:
        return []
    # simple approach: approximate tokens ~ characters. We'll split by sentences/lines for nicer chunks
    text = text.replace('\r\n', '\n')
    pieces = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        if end >= length:
            pieces.append(text[start: length])
            break
        # try to break at newline or space backwards
        sep_pos = text.rfind('\n', start, end)
        if sep_pos == -1:
            sep_pos = text.rfind(' ', start, end)
        if sep_pos == -1 or sep_pos <= start:
            sep_pos = end
        pieces.append(text[start:sep_pos])
        start = max(sep_pos - overlap, sep_pos)
    return [p.strip() for p in pieces if p.strip()]

# =========================
# OpenRouter client
# =========================
@st.cache_resource
def get_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("❌ Не найден OPENROUTER_API_KEY в st.secrets или окружении. Добавь ключ в Settings → Secrets.")
        st.stop()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client

client = get_client()

# =========================
# Optional local embedding model + FAISS vector store
# =========================
EMBEDDING_DIM = None
_embedding_model = None

def init_local_embedding_model(name="all-MiniLM-L6-v2"):
    global _embedding_model, EMBEDDING_DIM
    if not _HAS_SENT_TRANSFORMERS:
        st.warning("Локальные эмбеддинги недоступны — не установлена sentence-transformers. Установи дополнительные зависимости для векторной памяти.")
        return None
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(name)
        EMBEDDING_DIM = _embedding_model.get_sentence_embedding_dimension()
    return _embedding_model

def embed_texts(texts):
    """Возвращает list of vectors for given texts."""
    if _HAS_SENT_TRANSFORMERS:
        model = init_local_embedding_model()
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    else:
        raise RuntimeError("sentence-transformers не доступен")

# FAISS index holder in session
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "faiss_metadata" not in st.session_state:
    st.session_state.faiss_metadata = []

def create_faiss_index(vectors, metadatas):
    if not _HAS_FAISS:
        st.warning("FAISS не доступен — не установлена faiss-cpu. Установи опциональные зависимости.")
        return
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    st.session_state.faiss_index = index
    st.session_state.faiss_metadata = metadatas

def add_to_faiss(vectors, metadatas):
    """Append vectors and metadata to session faiss index (rebuilds if needed)."""
    import numpy as np
    if not _HAS_FAISS:
        raise RuntimeError("faiss не доступен")
    if st.session_state.faiss_index is None:
        create_faiss_index(vectors, metadatas)
    else:
        st.session_state.faiss_index.add(vectors)
        st.session_state.faiss_metadata.extend(metadatas)

def query_faiss(query_vec, top_k=5):
    import numpy as np
    if st.session_state.faiss_index is None:
        return []
    D, I = st.session_state.faiss_index.search(query_vec, top_k)
    results = []
    for idx in I[0]:
        if idx < len(st.session_state.faiss_metadata):
            results.append(st.session_state.faiss_metadata[idx])
    return results

# =========================
# UI and main logic
# =========================
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Sonoma, an experienced fullstack Python developer. "
        "Answer in Russian if user writes in Russian. Provide code blocks wrapped in ```python when relevant. "
        "Be helpful and concise."
    )
}

st.set_page_config(page_title="Sonoma Multi-Model Chat", layout="wide")
st.title("🤖 Sonoma / DeepSeek — расширенный Multi-Model Chat")

# Sidebar controls
with st.sidebar:
    st.header("Настройки чата")
    model_name = st.selectbox("Выберите модель", ["openrouter/sonoma-sky-alpha", "deepseek/deepseek-chat-v3.1:free"])
    mode = st.selectbox("Режим", ["balanced", "creative", "precise"])
    thinking_mode = st.checkbox("🧠 Режим максимального обдумывания (низкая температура)", value=False)
    use_vector_memory = st.checkbox("🗂️ Включить векторную память (RAG)", value=False)
    embedding_backend = st.selectbox("Backend для эмбеддингов (опционально)", ["local (sentence-transformers + faiss)", "qdrant", "pinecone", "none"])
    st.markdown("---")
    uploaded_files = st.file_uploader("📁 Загрузить файлы (pdf/csv/json/txt/images/py)", accept_multiple_files=True)
    if st.button("🗑️ Очистить чат и память"):
        st.session_state.clear()
        st.experimental_rerun()

# Ensure session messages
if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

# Process uploaded files
if uploaded_files:
    st.info(f"Загружено {len(uploaded_files)} файлов — обрабатываю...")
    for f in uploaded_files:
        if f.size > MAX_FILE_SIZE:
            st.warning(f"Файл {f.name} превышает {MAX_FILE_SIZE/1024/1024:.1f}MB и пропущен.")
            continue
        ext = f.name.split(".")[-1].lower()
        if ext not in SUPPORTED_EXT:
            st.warning(f"Формат {ext} не поддерживается, пропускаю {f.name}.")
            continue

        if ext in ["jpg", "jpeg", "png"]:
            img = Image.open(f)
            st.image(img, caption=f.name, width=300)
            preview = f"[IMAGE PREVIEW] {f.name} ({img.size[0]}x{img.size[1]})"
            st.session_state.messages.append({"role":"user","content": preview})
        elif ext == "pdf":
            try:
                pdf_bytes = f.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages_texts = []
                for p in reader.pages:
                    txt = p.extract_text() or ""
                    pages_texts.append(txt)
                full_text = "\n\n".join(pages_texts)
                st.text_area(f"Preview: {f.name}", value=(full_text[:1000] + "...") if len(full_text)>1000 else full_text, height=200)
                # chunk and optionally add to vector DB
                chunks = chunk_text(full_text)
                for c in chunks:
                    st.session_state.messages.append({"role":"user","content": f"[file:{f.name}] " + c})
                # optionally index chunks
                if use_vector_memory and embedding_backend.startswith("local") and _HAS_SENT_TRANSFORMERS and _HAS_FAISS:
                    vectors = embed_texts(chunks)
                    metadatas = [{"source": f.name, "text": c} for c in chunks]
                    add_to_faiss(vectors, metadatas)
                    st.success(f"Indexed {len(chunks)} chunks to local FAISS")
            except Exception as e:
                st.error(f"Ошибка чтения PDF {f.name}: {e}")
        elif ext in ["txt","py","md"]:
            text = f.read().decode("utf-8", errors="ignore")
            st.text_area(f"Preview: {f.name}", value=text[:2000], height=200)
            chunks = chunk_text(text)
            for c in chunks:
                st.session_state.messages.append({"role":"user","content": f"[file:{f.name}] " + c})
            if use_vector_memory and embedding_backend.startswith("local") and _HAS_SENT_TRANSFORMERS and _HAS_FAISS:
                vectors = embed_texts(chunks)
                metadatas = [{"source": f.name, "text": c} for c in chunks]
                add_to_faiss(vectors, metadatas)
                st.success(f"Indexed {len(chunks)} chunks to local FAISS")
        elif ext == "csv":
            try:
                df = pd.read_csv(f)
                st.dataframe(df.head(20))
                text = df.to_csv(index=False)
                chunks = chunk_text(text)
                for c in chunks:
                    st.session_state.messages.append({"role":"user","content": f"[file:{f.name}] " + c})
            except Exception as e:
                st.error(f"Ошибка парсинга CSV {f.name}: {e}")
        elif ext == "json":
            try:
                txt = f.read().decode("utf-8", errors="ignore")
                st.json(txt)
                chunks = chunk_text(txt)
                for c in chunks:
                    st.session_state.messages.append({"role":"user","content": f"[file:{f.name}] " + c})
            except Exception as e:
                st.error(f"Ошибка парсинга JSON {f.name}: {e}")

# Show chat history (user & assistant only)
st.write("### История чата")
for msg in st.session_state.messages[1:]:
    role = msg.get("role","user")
    content = msg.get("content","")
    if role == "user":
        st.markdown(f"**User:** {content}")
    elif role == "assistant":
        st.markdown(f"**Assistant:** {content}")
    else:
        # show system/file messages lightly
        st.markdown(f"*{role}*: {content[:400]}")

prompt = st.chat_input("Введите сообщение или команду (например: /summarize /search)")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    # убираем st.experimental_rerun()


# Command buttons
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Ответить (generate)"):
        # build messages and call API
        sys_prompt = SYSTEM_PROMPT.copy()
        if thinking_mode:
            sys_prompt["content"] += "\n\nPlease think step-by-step before answering."

        # retrieve vector context if enabled and available
        context_texts = []
        if use_vector_memory and _HAS_SENT_TRANSFORMERS and _HAS_FAISS and st.session_state.faiss_index is not None:
            # take last user message as query
            last_user = next((m for m in reversed(st.session_state.messages) if m["role"]=="user"), None)
            if last_user:
                q = last_user["content"]
                qv = embed_texts([q])
                hits = query_faiss(qv, top_k=5)
                if hits:
                    st.info(f"Найдено {len(hits)} релевантных фрагментов из памяти.")
                    for h in hits:
                        context_texts.append(h["text"] if "text" in h else h.get("content",""))
        # build API messages: system + optional retrieved + recent
        api_msgs = [sys_prompt]
        if context_texts:
            retrieved = "\n\n---\n".join(context_texts)
            api_msgs.append({"role":"system","content": f"Релевантные фрагменты из вашей базы знаний:\n{retrieved}"})
        # include recent history but limit length
        recent = st.session_state.messages[-MAX_HISTORY:]
        for m in recent[1:]:
            api_msgs.append({"role": m["role"], "content": m["content"]})

        # model params based on mode
        if mode == "creative":
            temp = 0.9
            max_tokens = 2000
        elif mode == "precise":
            temp = 0.15
            max_tokens = 1500
        else:
            temp = 0.5
            max_tokens = 1800
        if thinking_mode:
            temp = min(temp, 0.35)

        with st.spinner("Генерирую ответ..."):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=api_msgs,
                    temperature=temp,
                    max_tokens=max_tokens
                )
                reply = completion.choices[0].message.content
                st.session_state.messages.append({"role":"assistant","content":reply})
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Ошибка API: {e}")

with col2:
    if st.button("Summarize history"):
        # Summarize older part of history to compress memory
        if len(st.session_state.messages) <= 5:
            st.warning("История короткая — нечего суммаризовать.")
        else:
            older = st.session_state.messages[1: max(1, len(st.session_state.messages)-10)]
            old_texts = []
            for m in older:
                if isinstance(m.get("content"), str):
                    old_texts.append(f"{m['role']}: {m['content']}")
            summary_prompt = {"role":"system","content":"Суммаризуй следующие сообщения в 3-6 предложениях, выкопировав ключевые факты и предпочтения пользователя: " + "\n\n".join(old_texts)}
            try:
                summ = client.chat.completions.create(
                    model=model_name,
                    messages=[summary_prompt],
                    temperature=0.2,
                    max_tokens=400
                )
                summary_text = summ.choices[0].message.content
                # delete older chunk and insert summary as system context
                st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-10:]
                st.session_state.messages.insert(1, {"role":"system","content": f"Summary of earlier conversation: {summary_text}"})
                st.success("История суммаризована и вставлена как системное сообщение.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Ошибка суммаризации: {e}")

with col3:
    if st.button("Export chat as TXT"):
        if len(st.session_state.messages) <= 1:
            st.warning("Нет содержимого для экспорта.")
        else:
            out = io.StringIO()
            for m in st.session_state.messages[1:]:
                out.write(f"{m['role'].upper()}:\n{m['content']}\n\n")
            st.download_button("Download chat", data=out.getvalue(), file_name="chat_export.txt", mime="text/plain")

# Footer diagnostics
st.markdown("---")
st.markdown("**Diagnostics**")
st.write(f"Model: {model_name} | Mode: {mode} | Thinking: {thinking_mode}")
st.write(f"Messages in session: {len(st.session_state.messages)}")
if use_vector_memory:
    st.write(f"Local FAISS index present: {st.session_state.faiss_index is not None and _HAS_FAISS}")
    st.write(f"Sentence-transformers available: {_HAS_SENT_TRANSFORMERS}")
    st.write(f"faiss available: {_HAS_FAISS}")
    st.write(f"qdrant available: {_HAS_QDRANT}; pinecone available: {_HAS_PINECONE}")

st.caption("Подсказка: для векторной памяти включите local backend и установите sentence-transformers + faiss-cpu (см. requirements_optional.txt).")

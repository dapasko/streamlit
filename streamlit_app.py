# app.py
import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import pandas as pd
import mimetypes
import json
from typing import List, Dict, Any, Optional
import time
import base64
import os
import logging

# Logging
logging.basicConfig(level=logging.INFO)

# Установка tiktoken для точной оценки токенов (pip install tiktoken)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    st.warning("Установи tiktoken для точной оценки токенов: pip install tiktoken")

# =========================
# Конфиги
# =========================
MODEL_NAME = "openrouter/sonoma-sky-alpha"
MODEL_CONTEXT_TOKENS = 2_000_000
DEFAULT_MAX_TOKENS_RESPONSE = 200000
DEFAULT_TEMPERATURE = 0.7
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILES = 10

# =========================
# Утилиты
# =========================
def approx_tokens_from_text(text: str) -> int:
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            pass
    # Fallback
    if not text:
        return 0
    b = text.encode("utf-8")
    return max(1, int(len(b) / 4))

def safe_read_bytes(uploaded_file) -> bytes:
    uploaded_file.seek(0)
    return uploaded_file.read()

def text_download_link(text: str, filename: str, label: str = "Скачать"):
    b = text.encode("utf-8")
    href = f"data:file/txt;base64,{base64.b64encode(b).decode()}"
    st.markdown(f"[{label}]({href})", unsafe_allow_html=True)

# =========================
# Обработка файлов
# =========================
@st.cache_data
def process_file_bytes(name: str, file_bytes: bytes) -> Dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(name)
    extension = os.path.splitext(name)[1].lower()
    file_type = "unknown"
    content = ""
    preview = ""
    image_data = None  # Для миниатюр изображений (base64)
    size = len(file_bytes)

    if size > MAX_FILE_SIZE:
        return {"name": name, "content": f"[Файл {name} слишком большой: {size} bytes (>50MB)]", "type": "error", "preview": "", "size": size, "image_data": None}

    try:
        if extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
            file_type = "image"
            try:
                img = Image.open(io.BytesIO(file_bytes))
                content = f"[Image {name} — {img.size[0]}x{img.size[1]} px]"
                preview = content
                # Base64 для миниатюры
                buffered = io.BytesIO()
                img.thumbnail((200, 200))  # Масштабируем для preview
                img.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                content = f"[Image {name} — unable to open: {e}]"
                preview = content

        elif extension == ".pdf":
            file_type = "pdf"
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                text = "\n".join(text_parts)
                content = text
                preview = text[:500] + ("..." if len(text) > 500 else "")
            except Exception as e:
                content = f"[PDF {name} — error: {e}]"
                preview = content

        elif extension in [".txt", ".py", ".md", ".log"]:
            file_type = "text"
            content = file_bytes.decode("utf-8", errors="ignore")
            preview = content[:500] + ("..." if len(content) > 500 else "")

        elif extension == ".csv":
            file_type = "csv"
            text = file_bytes.decode("utf-8", errors="ignore")
            content = text
            try:
                df = pd.read_csv(io.StringIO(text))
                preview = f"CSV: {len(df)} rows, cols: {list(df.columns)}\n{df.head(3).to_csv(index=False)}"
            except:
                preview = text[:500] + ("..." if len(text) > 500 else "")

        elif extension == ".json":
            file_type = "json"
            text = file_bytes.decode("utf-8", errors="ignore")
            content = text
            try:
                data = json.loads(text)
                preview = json.dumps(data, ensure_ascii=False, indent=2)[:500] + ("..." if len(text) > 500 else "")
            except:
                preview = text[:500] + ("..." if len(text) > 500 else "")

        else:
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
                file_type = "text"
                content = text
                preview = text[:500] + ("..." if len(text) > 500 else "")
            except:
                file_type = "binary"
                content = f"[Binary {name}, {size} bytes]"
                preview = content

    except Exception as e:
        file_type = "error"
        content = f"[Error {name}: {e}]"
        preview = content

    return {
        "name": name, "content": content, "type": file_type, "preview": preview,
        "size": size, "image_data": image_data
    }

# =========================
# Построение сообщений и оценка
# =========================
def build_api_messages(system_prompt: str, chat_messages: List[Dict[str, Any]], files: List[Dict[str, Any]],
                       include_files: List[str], use_full_history: bool, limit_messages: int) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]

    for f in files:
        if f["name"] in include_files:
            preview = f.get("preview", "")
            messages.append({
                "role": "user",
                "content": f"[File: {f['name']} (type={f['type']}, size={f['size']} bytes)]\nPreview:\n{preview}\n\nFull:\n{f['content']}"
            })

    hist = chat_messages if use_full_history else chat_messages[-limit_messages:]
    for m in hist:
        messages.append({"role": m["role"], "content": m["content"]})

    return messages

def estimate_total_tokens(messages: List[Dict[str, str]]) -> int:
    total = sum(approx_tokens_from_text(m.get("content", "")) for m in messages)
    total += len(messages) * 20  # Overhead для ролей/metadata
    return total

# =========================
# Клиент OpenAI
# =========================
@st.cache_resource
def get_client(api_key: str) -> Optional[OpenAI]:
    if not api_key:
        return None
    try:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    except Exception as e:
        st.error(f"Ошибка клиента: {e}")
        return None

# =========================
# Инициализация session_state
# =========================
st.set_page_config(page_title="🤖 Sonoma Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# CSS стилизация
st.markdown("""
<style>
.main { background-color: #f0f2f6; padding: 1rem; }
.stChatMessage { padding: 1rem; border-radius: 15px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.user-bubble { background-color: #007bff; color: white; border-radius: 15px 15px 5px 15px; }
.assistant-bubble { background-color: #e9ecef; color: black; border-radius: 15px 15px 15px 5px; }
.sidebar .stButton > button { background-color: #28a745; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; }
.sidebar .stButton > button:hover { background-color: #218838; }
.token-ok { color: green; }
.token-warning { color: orange; font-weight: bold; }
.token-error { color: red; font-weight: bold; }
.file-preview { max-height: 150px; overflow-y: auto; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; font-size: 0.9em; }
.expander-header { font-weight: bold; color: #495057; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = []
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS_RESPONSE
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_TEMPERATURE
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are Sonoma, an experienced fullstack Python developer. "
        "You answer Russian or English requests with code, instructions, or file analysis. "
        "Be concise and provide examples when needed."
    )
if "use_full_history" not in st.session_state:
    st.session_state.use_full_history = True
if "limit_messages" not in st.session_state:
    st.session_state.limit_messages = 50
if "include_files" not in st.session_state:
    st.session_state.include_files = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Обрезаем историю, если слишком большая (>1000 сообщений)
if len(st.session_state.messages) > 1000:
    st.session_state.messages = st.session_state.messages[-500:]
    st.info("История обрезана для производительности.")

# =========================
# Sidebar рендер
# =========================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Настройки")


        # Модель и контекст (info)
        with st.expander("ℹ️ Инфо о модели", expanded=False):
            st.markdown(f"**Модель:** {MODEL_NAME}")
            st.markdown(f"**Контекст:** до {MODEL_CONTEXT_TOKENS:,} токенов")
            st.caption("Настройки ниже влияют на ответы. Для больших файлов — включай их по одному.")

        # Настройки слайдеры
        with st.expander("🎛️ Параметры генерации", expanded=True):
            st.session_state.max_tokens = st.slider(
                "📏 Max tokens (ответ)",
                min_value=256, max_value=MODEL_CONTEXT_TOKENS // 2, value=st.session_state.max_tokens, step=256
            )
            st.session_state.temperature = st.slider(
                "🎨 Temperature (креативность)",
                min_value=0.0, max_value=1.5, value=st.session_state.temperature, step=0.05
            )
            st.session_state.use_full_history = st.checkbox(
                "📚 Полная история", value=st.session_state.use_full_history
            )
            if not st.session_state.use_full_history:
                st.session_state.limit_messages = st.number_input(
                    "Последние N сообщений", min_value=1, max_value=500, value=st.session_state.limit_messages
                )

        # System prompt
        with st.expander("💬 System Prompt", expanded=False):
            st.session_state.system_prompt = st.text_area(
                "Системный промпт", value=st.session_state.system_prompt, height=100
            )

        # Файлы
        with st.expander("📁 Файлы (max 10)", expanded=True):
            if len(st.session_state.files) >= MAX_FILES:
                st.warning(f"Лимит: {MAX_FILES} файлов. Удали некоторые.")
            uploaded_files = st.file_uploader(
                "Загрузить", accept_multiple_files=True,
                type=['txt', 'pdf', 'jpg', 'jpeg', 'png', 'csv', 'json', 'py', 'md', 'log']
            )
            if uploaded_files:
                for uf in uploaded_files:
                    if len(st.session_state.files) >= MAX_FILES:
                        st.warning("Достигнут лимит файлов.")
                        break
                    try:
                        bytes_data = safe_read_bytes(uf)
                        if len(bytes_data) > MAX_FILE_SIZE:
                            st.error(f"{uf.name}: слишком большой (>50MB)")
                            continue
                        info = process_file_bytes(uf.name, bytes_data)
                        if not any(f["name"] == info["name"] for f in st.session_state.files):
                            info["include"] = True
                            st.session_state.files.append(info)
                            st.success(f"Загружен: {info['name']}")
                        else:
                            st.warning(f"{info['name']} уже есть.")
                    except Exception as e:
                        st.error(f"Ошибка {uf.name}: {e}")

            # Список файлов
            if st.session_state.files:
                st.subheader("Загруженные файлы")
                for i, f in enumerate(st.session_state.files):
                    cols = st.columns([5, 1, 1])
                    with cols[0]:
                        st.markdown(f"**{f['name']}** ({f['type']}, {f['size']/1024:.1f} KB)")
                        if f['type'] == 'image' and f.get('image_data'):
                            st.image(f"data:image/png;base64,{f['image_data']}", width=100, caption="Preview")
                        else:
                            st.caption(f["preview"][:200] + "..." if len(f["preview"]) > 200 else f["preview"], help="Короткий предпросмотр")
                    with cols[1]:
                        inc = st.checkbox("Вкл.", value=f.get("include", True), key=f"inc_{i}")
                        f["include"] = inc
                        if inc and f["name"] not in st.session_state.include_files:
                            st.session_state.include_files.append(f["name"])
                        elif not inc and f["name"] in st.session_state.include_files:
                            st.session_state.include_files.remove(f["name"])
                    with cols[2]:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.files.pop(i)
                            st.session_state.include_files = [n for n in st.session_state.include_files if n != f["name"]]
                            st.rerun()

        # Экспорт/Импорт
        with st.expander("💾 Экспорт/Импорт", expanded=False):
            if st.button("📤 Экспорт JSON"):
                data = {
                    "messages": st.session_state.messages,
                    "files": [{k: v for k, v in f.items() if k not in ["content", "image_data"]} for f in st.session_state.files],  # Без тяжёлого content
                    "system_prompt": st.session_state.system_prompt
                }
                st.download_button("Скачать JSON", json.dumps(data, ensure_ascii=False, indent=2), "chat.json", "application/json")

            if st.button("📥 TXT чата"):
                buf = io.StringIO()
                for m in st.session_state.messages:
                    role = "👤 User" if m["role"] == "user" else "🤖 Assistant"
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("ts", time.time())))
                    buf.write(f"{role} ({ts}):\n{m['content']}\n{'-'*80}\n\n")
                st.download_button("Скачать TXT", buf.getvalue(), "chat.txt", "text/plain")

            uploaded_chat = st.file_uploader("Импорт JSON", type=["json"])
            if uploaded_chat:
                try:
                    raw = safe_read_bytes(uploaded_chat).decode("utf-8")
                    obj = json.loads(raw)
                    if "messages" in obj:
                        st.session_state.messages = obj["messages"][-100:]  # Последние 100
                    if "files" in obj:
                        for imp in obj["files"]:
                            if len(st.session_state.files) < MAX_FILES and not any(f["name"] == imp.get("name") for f in st.session_state.files):
                                stub = {**imp, "content": f"[Content not imported for {imp.get('name')}]", "include": False, "image_data": None}
                                st.session_state.files.append(stub)
                    st.success("Импорт завершён.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка импорта: {e}")

        # Оценка токенов (перенесена сюда)
        with st.expander("📊 Статистика контекста", expanded=False):
            api_messages = build_api_messages(
                st.session_state.system_prompt, st.session_state.messages, st.session_state.files,
                st.session_state.include_files, st.session_state.use_full_history, st.session_state.limit_messages
            )
            approx_tokens = estimate_total_tokens(api_messages)
            max_allowed = MODEL_CONTEXT_TOKENS - st.session_state.max_tokens
            if approx_tokens > max_allowed:
                color = "token-error"
                msg = f"❌ ~{approx_tokens:,} токенов (превышает лимит {max_allowed:,} для input)"
            elif approx_tokens > max_allowed * 0.8:
                color = "token-warning"
                msg = f"⚠️ ~{approx_tokens:,} токенов (близко к лимиту {max_allowed:,})"
            else:
                color = "token-ok"
                msg = f"✅ ~{approx_tokens:,} токенов (ок, лимит {max_allowed:,})"
            st.markdown(f'<span class="{color}">{msg}</span>', unsafe_allow_html=True)
            if approx_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS:
                st.caption("Совет: Отключи файлы/историю или уменьши max_tokens.")

# =========================
# Основной чат
# =========================
def render_chat():
    st.title("🤖 Sonoma Chat")
    st.caption("Упрощённый чат с поддержкой файлов и большим контекстом. Настройки в сайдбаре.")

    # Верхние кнопки (компактно)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Очистить чат"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Очистить файлы"):
            st.session_state.files = []
            st.session_state.include_files = []
            st.rerun()

    # Вывод чата
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

    # Ввод
    user_prompt = st.chat_input("Сообщение (RU/EN)...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt, "ts": time.time()})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-bubble">{user_prompt}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            with st.spinner("Генерирую ответ..."):
                client = get_client(st.session_state.api_key)
                if client is None:
                    st.error("Укажи API ключ в сайдбаре.")
                    st.session_state.messages.append({"role": "assistant", "content": "❌ Нет API ключа.", "ts": time.time()})
                    st.rerun()

                # Сборка сообщений
                api_messages = build_api_messages(
                    st.session_state.system_prompt, st.session_state.messages, st.session_state.files,
                    st.session_state.include_files, st.session_state.use_full_history, st.session_state.limit_messages
                )
                approx_in = estimate_total_tokens(api_messages)
                if approx_in + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS:
                    if st.session_state.use_full_history:
                        st.error("Контекст слишком большой. Отключи полную историю в сайдбаре.")
                        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Превышен лимит — настрой сайдбар.", "ts": time.time()})
                        st.rerun()
                    else:
                        reduced = max(10, st.session_state.limit_messages // 2)
                        st.info(f"Обрезаю историю до {reduced} сообщений...")
                        api_messages = build_api_messages(
                            st.session_state.system_prompt, st.session_state.messages, st.session_state.files,
                            st.session_state.include_files, False, reduced
                        )
                        approx_in = estimate_total_tokens(api_messages)
                        if approx_in + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS:
                            st.error("Всё ещё превышает. Уменьши файлы/max_tokens.")
                            st.session_state.messages.append({"role": "assistant", "content": "❌ Лимит превышен.", "ts": time.time()})
                            st.rerun()

                # Стриминг запрос
                message_placeholder = st.empty()
                full_response = ""
                try:
                    stream = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(f'<div class="assistant-bubble">{full_response}▌</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response, "ts": time.time()})
                except Exception as e:
                    err = f"Ошибка: {e}"
                    message_placeholder.markdown(f'<div class="assistant-bubble" style="color:red;">{err}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": err, "ts": time.time()})
                st.rerun()

    # Нижние кнопки
    col3, col4, col5 = st.columns([1, 1, 1])
    with col3:
        if st.button("🔁 Повторить последний"):
            last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
            if last_user:
                st.session_state.messages.append({"role": "user", "content": last_user, "ts": time.time()})
                st.rerun()
            else:
                st.info("Нет сообщения для повтора.")
    with col4:
        if st.button("🗑️ Удалить последнее"):
            if st.session_state.messages:
                st.session_state.messages.pop()
                st.rerun()
    with col5:
        st.caption("Подсказка: Для анализа файлов — включи их в сайдбаре.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    render_sidebar()
    st.divider()
    render_chat()
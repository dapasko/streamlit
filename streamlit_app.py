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

# =========================
# Конфиги
# =========================
MODEL_NAME = "openrouter/sonoma-sky-alpha"
MODEL_CONTEXT_TOKENS = 2_000_000  # окно контекста модели (в токенах) — указано пользователем
DEFAULT_MAX_TOKENS_RESPONSE = 200000
DEFAULT_TEMPERATURE = 0.7

# =========================
# Утилиты
# =========================
def approx_tokens_from_text(text: str) -> int:
    """
    Примерная оценка количества токенов в тексте.
    Правило: ~4 символа на токен или ~1 token = 4 символа (приближённо).
    Более консервативная оценка: 1 токен ~ 4 байта символов ASCII.
    Это **приближённо**, но даёт представление.
    """
    if not text:
        return 0
    # учитываем, что некоторые символы (Unicode) могут занимать больше байт;
    # переводим в utf-8 длину в байтах
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
# Обработка файлов (функции)
# =========================
@st.cache_data
def process_file_bytes(name: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Обрабатываем файл по байтам и возвращаем структуру:
    {
        "name": str,
        "content": str,  # текстовое содержимое (для отправки в контекст)
        "type": str,     # csv, pdf, image, txt, json, py, unknown
        "preview": str,  # предпросмотр (короткий)
        "size": int
    }
    """
    mime_type, _ = mimetypes.guess_type(name)
    extension = os.path.splitext(name)[1].lower()
    file_type = "unknown"
    content = ""
    preview = ""
    size = len(file_bytes)

    try:
        if extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
            file_type = "image"
            try:
                img = Image.open(io.BytesIO(file_bytes))
                content = f"[Image {name} — {img.size[0]}x{img.size[1]} px]"
                preview = content
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
                preview = text[:1000] + ("..." if len(text) > 1000 else "")
            except Exception as e:
                content = f"[PDF {name} — error extracting text: {e}]"
                preview = content

        elif extension in [".txt", ".py", ".md", ".log"]:
            file_type = "text"
            content = file_bytes.decode("utf-8", errors="ignore")
            preview = content[:1000] + ("..." if len(content) > 1000 else "")

        elif extension == ".csv":
            file_type = "csv"
            text = file_bytes.decode("utf-8", errors="ignore")
            content = text
            try:
                df = pd.read_csv(io.StringIO(text))
                preview = f"CSV {name}: {len(df)} rows, columns: {list(df.columns)}\n" + df.head(5).to_csv(index=False)
            except Exception:
                preview = text[:1000] + ("..." if len(text) > 1000 else "")

        elif extension == ".json":
            file_type = "json"
            text = file_bytes.decode("utf-8", errors="ignore")
            content = text
            try:
                data = json.loads(text)
                preview = json.dumps(data, ensure_ascii=False, indent=2)[:1000] + ("..." if len(text) > 1000 else "")
            except Exception:
                preview = text[:1000] + ("..." if len(text) > 1000 else "")

        else:
            # неизвестный тип: пробуем прочитать как текст
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
                file_type = "text"
                content = text
                preview = text[:1000] + ("..." if len(text) > 1000 else "")
            except:
                file_type = "binary"
                content = f"[Бинарный файл {name}, размер {size} байт]"
                preview = content

    except Exception as e:
        file_type = "error"
        content = f"[Ошибка обработки файла {name}: {e}]"
        preview = content

    return {
        "name": name,
        "content": content,
        "type": file_type,
        "preview": preview,
        "size": size
    }

# =========================
# Построение сообщений для API
# =========================
def build_api_messages(system_prompt: str,
                       chat_messages: List[Dict[str, str]],
                       files: List[Dict[str, Any]],
                       include_files: List[str],
                       use_full_history: bool,
                       limit_messages: int) -> List[Dict[str, str]]:
    """
    Формируем список сообщений (system + user/assistant) для отправки в API.
    include_files — список имён файлов, которые включаем в контекст.
    use_full_history — если True, используем всю историю; иначе последние limit_messages.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Вставляем файлы как отдельные user-сообщения (в порядке загрузки)
    for f in files:
        if f["name"] in include_files:
            # поместим сначала короткую мета-информацию, затем полный контент
            preview = f.get("preview", "")
            # При слишком большом размере — можно оставить только preview и ссылку на скачивание
            messages.append({
                "role": "user",
                "content": f"[Файл: {f['name']} (type={f['type']}, size={f['size']} bytes)]\nPreview:\n{preview}\n\nFull content:\n{f['content']}"
            })

    # Добавляем историю
    hist = chat_messages if use_full_history else chat_messages[-limit_messages:]
    for m in hist:
        messages.append({"role": m["role"], "content": m["content"]})

    return messages

def estimate_total_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Простая оценка: суммируем приближённые токены для всех сообщений.
    """
    total = 0
    for m in messages:
        total += approx_tokens_from_text(m.get("content", ""))
    # плюс немного служебных токенов
    total += int(len(messages) * 3)
    return total

# =========================
# Инициализация Streamlit session_state
# =========================
st.set_page_config(page_title="🤖 Мульти-Чат (Sonoma)", page_icon="🤖", layout="wide")

# Custom CSS для мобильного и темной темы (улучшенная версия)
st.markdown("""
<style>
/* ===== Общие настройки ===== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e4e6eb;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

/* ===== Чат ===== */
.stChatMessage {
    border-radius: 16px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.4;
}

/* Пользователь */
.stChatMessage[data-testid="stChatMessage-user"] {
    background: #2563eb;
    color: white;
    margin-left: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}

/* Ассистент */
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background: #1f2937;
    color: #e5e7eb;
    margin-right: auto;
    border: 1px solid #374151;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

/* ===== Sidebar ===== */
.stSidebar {
    background: #111827 !important;
    color: #e5e7eb !important;
    border-right: 1px solid #1f2937;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label {
    color: #f3f4f6 !important;
}

/* ===== Карточки (expanders) ===== */
.stExpander {
    border-radius: 12px;
    border: 1px solid #334155;
    background: #1e293b;
    margin-bottom: 8px !important;
}
.stExpander > div > div {
    background: transparent !important;
}

/* ===== Кнопки ===== */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: 0.2s;
    border: none;
    padding: 0.6rem 1rem;
}
.stButton > button:hover {
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: scale(0.98);
}

/* Цветные кнопки */
.stButton > button[kind="primary"] {
    background: #2563eb;
    color: white;
}
.stButton > button[kind="secondary"] {
    background: #475569;
    color: #f9fafb;
}

/* ===== Inputs ===== */
.stTextArea textarea, .stTextInput input {
    border-radius: 8px;
    border: 1px solid #374151;
    background: #111827;
    color: #e5e7eb;
}
.stSlider > div > div > div {
    background: #2563eb !important;
}

/* ===== Footer ===== */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []  # список dict {"role": "user"/"assistant", "content": "...", "ts": ...}
if "files" not in st.session_state:
    st.session_state.files = []  # список dict возвращаемых process_file_bytes + {"include": True}
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
    st.session_state.include_files = []  # имена файлов включённые в контекст
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets.get("OPENROUTER_API_KEY") or ""

# =========================
# Sidebar — настройки (улучшенная версия, все в компактных блоках)
# =========================
with st.sidebar:
    st.header("⚙️ Настройки")

    # Основные настройки (видны всегда, компактно)
    st.markdown("**Модель**: " + MODEL_NAME)
    st.caption("Окно контекста: **2 000 000** токенов.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.max_tokens = st.slider(
            "📏 Max tokens (ответ)",
            min_value=256,
            max_value=MODEL_CONTEXT_TOKENS,
            value=st.session_state.max_tokens,
            step=256,
            help="Макс. токенов в ответе. Input + output ≤ 2M."
        )
    with col2:
        st.session_state.temperature = st.slider(
            "🎨 Temperature",
            min_value=0.0,
            max_value=1.5,
            value=st.session_state.temperature,
            step=0.05,
            help="0.0 — точные, 1.5 — креативные ответы."
        )

    # Режим истории (чекбокс + input, компактно)
    col_hist1, col_hist2 = st.columns([3, 2])
    with col_hist1:
        st.session_state.use_full_history = st.checkbox(
            "📚 Вся история в контексте",
            value=st.session_state.use_full_history
        )
    with col_hist2:
        if not st.session_state.use_full_history:
            st.session_state.limit_messages = st.number_input(
                "Последние N сообщений",
                min_value=1, max_value=5000, value=st.session_state.limit_messages,
                help="Ограничение истории для экономии токенов."
            )

    # Разделитель минимальный
    st.markdown("---")

    # Файлы — в expander (сворачивается)
    with st.expander("📁 Файлы (загрузка и управление)", expanded=False):
        st.caption("Загружай файлы — выбирай, какие включать в контекст.")

        uploaded_files = st.file_uploader(
            "📁 Загрузить файлы (multi)",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'jpg', 'jpeg', 'png', 'csv', 'json', 'py', 'md', 'log'],
            help="Поддержка: текст, PDF, изображения, CSV, JSON, Python, Markdown. Макс. 50MB/файл."
        )

        if uploaded_files:
            for uf in uploaded_files:
                try:
                    bytes_data = safe_read_bytes(uf)
                    info = process_file_bytes(uf.name, bytes_data)
                    if not any(f["name"] == info["name"] for f in st.session_state.files):
                        info["include"] = True
                        st.session_state.files.append(info)
                        st.success(f"Загружен: {info['name']}")
                    else:
                        st.warning(f"Файл {info['name']} уже загружен.")
                except Exception as e:
                    st.error(f"Ошибка загрузки {uf.name}: {e}")

        # Список файлов (компактный)
        if st.session_state.files:
            for i, f in enumerate(st.session_state.files):
                cols = st.columns([5, 1, 1])  # Уменьшил ширину preview
                with cols[0]:
                    st.markdown(f"**{f['name']}** ({f['type']}, {f['size']} bytes)")
                    st.text_area(f"preview_{i}", value=f.get("preview", "")[:1000], height=60, key=f"preview_{i}", label_visibility="collapsed")  # Скрыл лейбл, уменьшил высоту
                with cols[1]:
                    inc = st.checkbox("Включить", value=f.get("include", True), key=f"inc_{i}")
                    st.session_state.files[i]["include"] = inc
                    if inc and f["name"] not in st.session_state.include_files:
                        st.session_state.include_files.append(f["name"])
                    if not inc and f["name"] in st.session_state.include_files:
                        st.session_state.include_files.remove(f["name"])
                with cols[2]:
                    if st.button("❌", key=f"del_{i}", help="Удалить"):
                        st.session_state.files.pop(i)
                        st.session_state.include_files = [n for n in st.session_state.include_files if n != f["name"]]
                        st.rerun()

    # Минимальный разделитель
    st.markdown("---")

    # Системный промпт — в expander
    with st.expander("🔧 Системный промпт", expanded=False):
        st.session_state.system_prompt = st.text_area(
            label="System prompt (как system message)",
            value=st.session_state.system_prompt,
            height=100,  # Уменьшил высоту
            help="По умолчанию: 'You are Sonoma, an experienced fullstack Python developer...'"
        )

    # Минимальный разделитель
    st.markdown("---")

    # Экспорт/импорт — в expander
    with st.expander("💾 Экспорт/импорт чата", expanded=False):
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("🔁 Экспорт (JSON)"):
                data = {
                    "messages": st.session_state.messages,
                    "files": [{k: v for k, v in f.items() if k != "content"} for f in st.session_state.files],
                    "system_prompt": st.session_state.system_prompt
                }
                st.download_button(
                    label="Скачать JSON",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name="chat_export.json",
                    mime="application/json"
                )
        with col_exp2:
            uploaded_chat = st.file_uploader("Импорт (JSON)", type=["json"], key="upload_chat")
            if uploaded_chat:
                try:
                    raw = safe_read_bytes(uploaded_chat).decode("utf-8", errors="ignore")
                    obj = json.loads(raw)
                    if "messages" in obj:
                        st.session_state.messages = obj["messages"]
                    if "files" in obj and isinstance(obj["files"], list):
                        for imported in obj["files"]:
                            if not any(f["name"] == imported.get("name") for f in st.session_state.files):
                                stub = {
                                    "name": imported.get("name"),
                                    "content": imported.get("content", f"[content not available for {imported.get('name')}]"),
                                    "type": imported.get("type", "unknown"),
                                    "preview": imported.get("preview", ""),
                                    "size": imported.get("size", 0),
                                    "include": False
                                }
                                st.session_state.files.append(stub)
                    st.success("Чат импортирован (файлы — как метаданные).")
                    st.rerun()
                except Exception as e:
                    st.error("Ошибка импорта: " + str(e))

    # Минимальный разделитель
    st.markdown("---")

    # Управление чатом — в expander (улучшенный вид: компактные кнопки, инфо в markdown)
    with st.expander("🛠️ Управление чатом", expanded=False):
        bot_cols = st.columns(3)  # Убрал [1,1,1] на равные
        with bot_cols[0]:
            if st.button("🔁 Повторить последний"):
                last_user = None
                for m in reversed(st.session_state.messages):
                    if m["role"] == "user":
                        last_user = m["content"]
                        break
                if last_user:
                    st.session_state.messages.append({"role": "user", "content": last_user, "ts": time.time()})
                    st.rerun()
                else:
                    st.info("Нет пользовательского сообщения.")
        with bot_cols[1]:
            if st.button("💾 Сохранить (TXT)"):
                buf = io.StringIO()
                for m in st.session_state.messages:
                    role = "User" if m["role"] == "user" else "Assistant"
                    ts = m.get("ts", "")
                    buf.write(f"{role} ({ts}):\n{m['content']}\n\n")
                st.download_button(
                    label="Скачать TXT",
                    data=buf.getvalue(),
                    file_name="chat.txt",
                    mime="text/plain"
                )
        with bot_cols[2]:
            if st.button("📥 Очистить последний"):
                if st.session_state.messages:
                    st.session_state.messages.pop()
                    st.rerun()

        # Инфо модели: в компактном markdown
        st.markdown("**Инфо модели:**")
        st.markdown("""
        - **Модель**: openrouter/sonoma-sky-alpha  
        - **Контекст**: до 2M токенов  
        - **Настройка**: max tokens/temperature для длины/креативности.  
        - **Файлы**: Включай по отдельности для больших объёмов.
        """)

# =========================
# Создаём клиента OpenAI (OpenRouter)
# =========================
def get_client(api_key: str) -> Optional[OpenAI]:
    if not api_key:
        return None
    try:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    except Exception as e:
        st.error(f"Не удалось создать OpenAI клиент: {e}")
        return None

client = get_client(st.session_state.api_key)

# =========================
# Основная область — чат
# =========================
st.title("🤖 Мульти-Чат (Sonoma)")
st.caption("Модель: openrouter/sonoma-sky-alpha — окно контекста до 2M токенов. Настройки в боковой панели.")

# Панель управления чатом: кнопки (компактно)
top_cols = st.columns([1, 1, 4])
with top_cols[0]:
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.rerun()
with top_cols[1]:
    if st.button("🗑️ Стереть файлы"):
        st.session_state.files = []
        st.session_state.include_files = []
        st.rerun()
with top_cols[2]:
    # индикатор токенов
    api_messages_preview = build_api_messages(
        st.session_state.system_prompt,
        st.session_state.messages,
        st.session_state.files,
        st.session_state.include_files,
        st.session_state.use_full_history,
        st.session_state.limit_messages
    )
    approx_tokens = estimate_total_tokens(api_messages_preview)
    st.markdown(f"**Токены контекста:** ~{approx_tokens:,} (макс. {MODEL_CONTEXT_TOKENS:,})")
    if approx_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS * 1.1:
        st.error("⚠️ Контекст + ответ превышают 2M токенов. Уменьши историю/файлы/max_tokens.")

# Вывод сообщений
chat_box = st.container()
with chat_box:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# Ввод сообщения
user_prompt = st.chat_input("Введите сообщение (русский/английский)...")
if user_prompt and user_prompt.strip():
    st.session_state.messages.append({"role": "user", "content": user_prompt, "ts": time.time()})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Отправляю запрос...")

        api_messages = build_api_messages(
            st.session_state.system_prompt,
            st.session_state.messages,
            st.session_state.files,
            st.session_state.include_files,
            st.session_state.use_full_history,
            st.session_state.limit_messages
        )

        approx_in_tokens = estimate_total_tokens(api_messages)
        if approx_in_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS * 1.1:
            if st.session_state.use_full_history:
                placeholder.markdown(
                    "⚠️ Контекст слишком большой. Отключи 'Вся история' или уменьши файлы."
                )
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ Запрос отменён — превышен лимит."})
            else:
                reduced = st.session_state.limit_messages // 2
                placeholder.markdown(f"⚠️ Обрезаю историю до {reduced} сообщений...")
                api_messages = build_api_messages(
                    st.session_state.system_prompt,
                    st.session_state.messages,
                    st.session_state.files,
                    st.session_state.include_files,
                    False,
                    reduced
                )
                approx_in_tokens = estimate_total_tokens(api_messages)
                if approx_in_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS * 1.1:
                    placeholder.markdown("❌ Всё ещё превышает лимит. Уменьши файлы/max_tokens.")
                    st.session_state.messages.append({"role": "assistant", "content": "❌ Запрос отменён — превышен лимит."})
                    st.rerun()
        else:
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=float(st.session_state.temperature),
                    max_tokens=int(st.session_state.max_tokens)
                )
                reply = resp.choices[0].message.content if resp.choices else str(resp)
                placeholder.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply, "ts": time.time()})
            except Exception as e:
                err_text = f"⚠️ Ошибка модели: {e}"
                placeholder.markdown(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})

# Footer
st.markdown("---")
st.caption("💡 Совет: Для больших файлов (PDF/CSV/JSON) включай их по отдельности, чтобы не превысить лимит токенов.")
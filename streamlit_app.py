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

import re

def render_message(role: str, content: str):
    """
    Красиво отображает сообщение чата (user/assistant).
    Если есть блоки кода, выводим их отдельно с подсветкой и кнопкой копирования.
    """
    bg_color = "#f0f2f6" if role == "user" else "#e8f5e9"  # разные цвета для user/assistant

    with st.container():
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 8px; background-color: {bg_color}; margin-bottom: 10px;">
            <b>{'👤 Пользователь' if role == 'user' else '🤖 Ассистент'}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Разбираем текст на блоки: обычный текст и код
        code_pattern = r"```([\w]*)\n(.*?)```"
        parts = re.split(code_pattern, content, flags=re.DOTALL)

        # parts идёт как [текст, lang, code, текст, lang, code, ...]
        i = 0
        while i < len(parts):
            if i + 2 < len(parts) and parts[i+1] != "":
                # Это блок кода
                lang = parts[i+1]
                code = parts[i+2]
                if parts[i].strip():
                    st.markdown(parts[i])  # текст до кода
                st.code(code, language=lang if lang else None)
                i += 3
            else:
                if parts[i].strip():
                    st.markdown(parts[i])
                i += 1

        # Кнопка копирования всего сообщения
        st.button(
            "📋 Скопировать сообщение",
            key=f"copy_{role}_{hash(content)}",
            on_click=lambda c=content: st.session_state.update({"_clipboard": c}),
            use_container_width=True
        )


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

    # --- Основные настройки ---
    st.markdown(f"**Модель:** `{MODEL_NAME}`")
    st.caption(f"Контекст: до **{MODEL_CONTEXT_TOKENS:,}** токенов")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.max_tokens = st.number_input(
            "📏 Max tokens",
            min_value=256,
            max_value=MODEL_CONTEXT_TOKENS,
            value=st.session_state.max_tokens,
            step=256,
        )
    with col2:
        st.session_state.temperature = st.slider(
            "🎨 Temp",
            min_value=0.0,
            max_value=1.5,
            value=st.session_state.temperature,
            step=0.05,
        )

    colh1, colh2 = st.columns([3, 2])
    with colh1:
        st.session_state.use_full_history = st.checkbox(
            "📚 Вся история", value=st.session_state.use_full_history
        )
    with colh2:
        if not st.session_state.use_full_history:
            st.session_state.limit_messages = st.number_input(
                "N сообщений",
                min_value=1,
                max_value=5000,
                value=st.session_state.limit_messages,
            )

    st.markdown("---")

    # --- Файлы ---
    with st.expander("📂 Файлы", expanded=False):
        uploaded_files = st.file_uploader(
            "Загрузить файлы",
            type=['txt','pdf','jpg','jpeg','png','csv','json','py','md','log'],
            accept_multiple_files=True,
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

        for i, f in enumerate(st.session_state.files):
            with st.expander(f"📄 {f['name']} ({f['type']}, {f['size']}b)"):
                st.checkbox(
                    "Включить в контекст",
                    value=f.get("include", True),
                    key=f"inc_{i}"
                )
                st.text_area(
                    "preview",
                    value=f.get("preview", "")[:800],
                    height=100,
                    label_visibility="collapsed"
                )
                if st.button("❌ Удалить", key=f"del_{i}"):
                    st.session_state.files.pop(i)
                    st.session_state.include_files = [
                        n for n in st.session_state.include_files if n != f["name"]
                    ]
                    st.rerun()

    # --- Системный промпт ---
    with st.expander("📝 Системный промпт", expanded=False):
        st.session_state.system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=120,
        )

    # --- Экспорт / Импорт ---
    with st.expander("💾 Экспорт / Импорт", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Экспорт JSON"):
                export_data = {
                    "messages": st.session_state.messages,
                    "files": [{k: v for k, v in f.items() if k != "content"} for f in st.session_state.files],
                    "system_prompt": st.session_state.system_prompt,
                }
                st.download_button(
                    "Скачать",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name="chat.json",
                    mime="application/json",
                )
        with col2:
            uploaded_chat = st.file_uploader("Импорт JSON", type="json")
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

    # --- Управление чатом ---
    with st.expander("🛠️ Управление чатом", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁 Повторить последний"):
                last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
                if last_user:
                    st.session_state.messages.append({"role": "user", "content": last_user, "ts": time.time()})
                    st.rerun()
        with col2:
            if st.button("📥 Очистить последний"):
                if st.session_state.messages:
                    st.session_state.messages.pop()
                    st.rerun()
        st.button("🗑️ Очистить весь чат", on_click=lambda: st.session_state.update({"messages": []}))
        st.button("🗑️ Удалить все файлы", on_click=lambda: st.session_state.update({"files": [], "include_files": []}))

    # --- Инфо о модели ---
    st.info(
        "🤖 **Модель**: openrouter/sonoma-sky-alpha\n"
        f"📏 Контекст: до {MODEL_CONTEXT_TOKENS:,} токенов\n"
        "📂 Файлы: включай выборочно, чтобы экономить контекст"
    )


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
        render_message(msg["role"], msg["content"])


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
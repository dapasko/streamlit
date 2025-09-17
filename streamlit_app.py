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
# Sidebar — настройки
# =========================
with st.sidebar:
    st.header("⚙️ Настройки")

    st.markdown("**Модель**: " + MODEL_NAME)
    st.write("Окно контекста модели: **2 000 000** токенов (приближённо).")

    # API ключ
    api_key_input = st.text_input("API-ключ OpenRouter (OPENROUTER_API_KEY)", value=st.session_state.api_key, type="password")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    # max tokens (ответ)
    st.session_state.max_tokens = st.slider(
        "📏 Макс. токенов для ответа",
        min_value=256,
        max_value=MODEL_CONTEXT_TOKENS,
        value=st.session_state.max_tokens,
        step=256,
        help="Максимальное число токенов в *ответе*. Учти, что input + output ≤ 2М."
    )

    # temperature
    st.session_state.temperature = st.slider(
        "🎨 Креативность (temperature)",
        min_value=0.0,
        max_value=1.5,
        value=st.session_state.temperature,
        step=0.05,
        help="0.0 — максимально точные/детерминированные ответы, 1.5 — более креативно."
    )

    # режим истории
    st.session_state.use_full_history = st.checkbox(
        "📚 Использовать всю историю чата в контексте",
        value=st.session_state.use_full_history
    )
    if not st.session_state.use_full_history:
        st.session_state.limit_messages = st.number_input(
            "Последние N сообщений (если не использовать всю историю)",
            min_value=1, max_value=5000, value=st.session_state.limit_messages
        )

    st.markdown("---")
    st.subheader("Файлы")
    st.write("Загружай файлы — можно выбрать, какие из них включать в контекст.")

    uploaded_files = st.file_uploader(
        "📁 Загрузить файлы (multi)",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'jpg', 'jpeg', 'png', 'csv', 'json', 'py', 'md', 'log'],
        help="Поддерживаемые: текст, PDF, изображения, CSV, JSON, Python, Markdown. Макс. размер: 50MB на файл (строго проверять не будем)."
    )

    if uploaded_files:
        for uf in uploaded_files:
            try:
                bytes_data = safe_read_bytes(uf)
                info = process_file_bytes(uf.name, bytes_data)
                # добавляем только если ещё нет файла с таким именем
                if not any(f["name"] == info["name"] for f in st.session_state.files):
                    info["include"] = True
                    st.session_state.files.append(info)
                    st.success(f"Загружен: {info['name']}")
                else:
                    st.warning(f"Файл {info['name']} уже загружен.")
            except Exception as e:
                st.error(f"Ошибка загрузки {uf.name}: {e}")

    # показать список загруженных файлов с макетом включения/удаления
    if st.session_state.files:
        for i, f in enumerate(st.session_state.files):
            cols = st.columns([6, 1, 1])
            with cols[0]:
                st.markdown(f"**{f['name']}** ({f['type']}, {f['size']} bytes)")
                st.text_area(f"preview_{i}", value=f.get("preview", "")[:2000], height=80, key=f"preview_{i}")
            with cols[1]:
                inc = st.checkbox("Включить", value=f.get("include", True), key=f"inc_{i}")
                st.session_state.files[i]["include"] = inc
                if inc and f["name"] not in st.session_state.include_files:
                    st.session_state.include_files.append(f["name"])
                if not inc and f["name"] in st.session_state.include_files:
                    st.session_state.include_files.remove(f["name"])
            with cols[2]:
                if st.button("Удалить", key=f"del_{i}"):
                    st.session_state.files.pop(i)
                    # синхронизировать include_files
                    st.session_state.include_files = [n for n in st.session_state.include_files if n != f["name"]]
                    st.experimental_rerun()

    st.markdown("---")
    st.subheader("Системный промпт")
    st.session_state.system_prompt = st.text_area("System prompt (используется как system message)", value=st.session_state.system_prompt, height=120)

    st.markdown("---")
    st.write("Экспорт/импорт чата")
    if st.button("🔁 Экспортировать чат (JSON)"):
        data = {
            "messages": st.session_state.messages,
            "files": [{k: v for k, v in f.items() if k != "content"} for f in st.session_state.files],
            "system_prompt": st.session_state.system_prompt
        }
        st.download_button("Скачать JSON", data=json.dumps(data, ensure_ascii=False, indent=2), file_name="chat_export.json")

    uploaded_chat = st.file_uploader("Импортировать чат (JSON)", type=["json"], key="upload_chat")
    if uploaded_chat:
        try:
            raw = safe_read_bytes(uploaded_chat).decode("utf-8", errors="ignore")
            obj = json.loads(raw)
            if "messages" in obj:
                st.session_state.messages = obj["messages"]
            if "files" in obj and isinstance(obj["files"], list):
                # при импорте файлов — метаданные
                for imported in obj["files"]:
                    if not any(f["name"] == imported.get("name") for f in st.session_state.files):
                        # добавим заглушку без content (пользователь может загрузить файл вновь)
                        stub = {
                            "name": imported.get("name"),
                            "content": imported.get("content", f"[content not available for {imported.get('name')}]"),
                            "type": imported.get("type", "unknown"),
                            "preview": imported.get("preview", ""),
                            "size": imported.get("size", 0),
                            "include": False
                        }
                        st.session_state.files.append(stub)
                st.success("Чат импортирован (файлы добавлены как метаданные).")
        except Exception as e:
            st.error("Ошибка импорта: " + str(e))

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
if client is None:
    st.warning("Укажите корректный OPENROUTER_API_KEY в сайдбаре, чтобы отправлять запросы к модели.")

# =========================
# Основная область — чат
# =========================
st.title("🤖 Мульти-Чат (Sonoma) — упрощённая и улучшенная версия")
st.caption("Модель: openrouter/sonoma-sky-alpha — окно контекста до 2M токенов. Все настройки слева.")

# Панель управления чатом: кнопки очистки, экспорт
top_cols = st.columns([1, 1, 4])
with top_cols[0]:
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.experimental_rerun()
with top_cols[1]:
    if st.button("🗑️ Стереть файлы"):
        st.session_state.files = []
        st.session_state.include_files = []
        st.experimental_rerun()
with top_cols[2]:
    # индикатор примерной оценки токенов для текущего контекста
    api_messages_preview = build_api_messages(
        st.session_state.system_prompt,
        st.session_state.messages,
        st.session_state.files,
        st.session_state.include_files,
        st.session_state.use_full_history,
        st.session_state.limit_messages
    )
    approx_tokens = estimate_total_tokens(api_messages_preview)
    st.markdown(f"**Оценка токенов входного контекста:** ~**{approx_tokens:,}** токенов.  (макс {MODEL_CONTEXT_TOKENS:,})")
    if approx_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS:
        st.error("Внимание: входной контекст + максимальные токены ответа превышают окно модели (2M). Уменьши историю/файлы или max_tokens.")

# Вывод предыдущих сообщений в виде чата
chat_box = st.container()
with chat_box:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# =========================
# Ввод нового сообщения
# =========================
user_prompt = st.chat_input("Введите сообщение (русский/английский)...")
if user_prompt:
    # добавляем сообщение в историю
    st.session_state.messages.append({"role": "user", "content": user_prompt, "ts": time.time()})
    # отобразить сразу как пользователь
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # создаём placeholder для ответа
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Отправляю запрос в модель...")

        # Сборка сообщений для API
        api_messages = build_api_messages(
            st.session_state.system_prompt,
            st.session_state.messages,
            st.session_state.files,
            st.session_state.include_files,
            st.session_state.use_full_history,
            st.session_state.limit_messages
        )

        # Оценка токенов
        approx_in_tokens = estimate_total_tokens(api_messages)
        if approx_in_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS:
            # пробуем автоматически сократить: если не используем всю историю — сократить до limit_messages
            if st.session_state.use_full_history:
                # предупредить и не отправлять
                placeholder.markdown(
                    "⚠️ Размер контекста слишком большой (вместе с ожидаемым ответом превышает окно модели в 2M токенов). "
                    "Включен режим: использовать всю историю. Пожалуйста, отключи 'Использовать всю историю' в сайдбаре или уменьшь количество файлов/их включение."
                )
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ Запрос не отправлен — контекст превышает лимит модели."})
            else:
                # пытаемся обрезать историю до половины limit_messages
                reduced = st.session_state.limit_messages // 2
                placeholder.markdown(
                    f"⚠️ Контекст слишком большой — автоматически обрезаю историю до последних {reduced} сообщений и повторяю попытку..."
                )
                api_messages = build_api_messages(
                    st.session_state.system_prompt,
                    st.session_state.messages,
                    st.session_state.files,
                    st.session_state.include_files,
                    False,
                    reduced
                )
                approx_in_tokens = estimate_total_tokens(api_messages)
                # если всё ещё слишком большой — отказываемся
                if approx_in_tokens + st.session_state.max_tokens > MODEL_CONTEXT_TOKENS:
                    placeholder.markdown(
                        "❌ Всё ещё превышает лимит. Уменьши число файлов, отключи некоторые файлы из контекста, либо уменьшай max_tokens."
                    )
                    st.session_state.messages.append({"role": "assistant", "content": "❌ Запрос не отправлен — контекст превышает лимит модели."})
                    st.experimental_rerun()

        # Если клиент не настроен
        if client is None:
            placeholder.markdown("⚠️ Клиент не настроен: укажите OPENROUTER_API_KEY в сайдбаре.")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ Клиент не настроен — укажите API ключ."})
        else:
            try:
                # Отправляем запрос
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=api_messages,
                    temperature=float(st.session_state.temperature),
                    max_tokens=int(st.session_state.max_tokens)
                )
                # Берём ответ
                # структура: completion.choices[0].message.content
                reply = ""
                try:
                    reply = resp.choices[0].message.content
                except Exception:
                    # если структура иная
                    reply = str(resp)
                # показываем и сохраняем
                placeholder.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply, "ts": time.time()})
            except Exception as e:
                err_text = f"⚠️ Ошибка при вызове модели: {e}"
                placeholder.markdown(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})

# =========================
# Нижняя панель: функции управления чатом
# =========================
st.markdown("---")
bot_cols = st.columns([1, 1, 1, 2])
with bot_cols[0]:
    if st.button("🔁 Повторить последний ответ"):
        # попытаемся повторно послать последний пользовательский запрос (если есть)
        last_user = None
        for m in reversed(st.session_state.messages):
            if m["role"] == "user":
                last_user = m["content"]
                break
        if last_user:
            # удалим последний ассистентский ответ, если есть
            # добавим копию user в конец и перезапустим (упрощённый путь: имитируем ввод)
            st.session_state.messages.append({"role": "user", "content": last_user, "ts": time.time()})
            st.experimental_rerun()
        else:
            st.info("Нет пользовательского сообщения для повтора.")
with bot_cols[1]:
    if st.button("💾 Сохранить чат (TXT)"):
        buf = io.StringIO()
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else "Assistant"
            ts = m.get("ts", "")
            buf.write(f"{role} ({ts}):\n{m['content']}\n\n")
        st.download_button("Скачать TXT", data=buf.getvalue(), file_name="chat.txt", mime="text/plain")
with bot_cols[2]:
    if st.button("📥 Очистить последний"):
        if st.session_state.messages:
            st.session_state.messages.pop()
            st.experimental_rerun()
with bot_cols[3]:
    st.markdown("**Информация модели / подсказки**")
    st.write(
        " - Модель: openrouter/sonoma-sky-alpha\n"
        " - Контекст до 2M токенов (приблизительно)\n"
        " - Если ответы кажутся слишком длинными/короткими — настраивай Max tokens и Temperature.\n"
        " - Для анализа больших файлов включай их в контекст (чекбокс 'Включить' в сайдбаре)."
    )

# =========================
# Footer: подсказки
# =========================
st.markdown("---")
st.caption("Совет: если используешь большие PDF/CSV/JSON, включай их по отдельности в контекст — иначе суммарный объём текста может превысить окно модели.")


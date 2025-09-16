import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import pandas as pd
import mimetypes
import json
from typing import List, Dict, Any, Optional

# =========================
# Настройка страницы
# =========================
st.set_page_config(page_title="🤖 Мульти-Чат", page_icon="🤖", layout="wide")

st.title("🤖 Мульти-Чат: Sonoma & DeepSeek")
st.caption("Обращайся к любой из моделей. Русский/английский язык поддерживается. Макс. токены: до 4096 для длинных ответов.")

# =========================
# Секреты и API
# =========================
def get_client(api_key: str) -> OpenAI:
    """Создаёт клиента OpenAI для OpenRouter."""
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

API_KEY = st.secrets.get("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("❌ API-ключ не найден. Добавьте OPENROUTER_API_KEY в secrets.toml.")
    st.stop()

client = get_client(API_KEY)

# =========================
# Системные промпты
# =========================
SYSTEM_PROMPTS = {
    "sonoma": {
        "role": "system",
        "content": (
            "You are Sonoma, an experienced fullstack Python developer. "
            "You answer Russian or English requests with code, instructions, or file analysis."
        )
    },
    "deepseek": {
        "role": "system",
        "content": (
            "You are DeepSeek, a smart assistant. Answer clearly and concisely. "
            "Support file analysis and general knowledge."
        )
    }
}

# =========================
# Инициализация сессии
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "sonoma"
if "files" not in st.session_state:
    st.session_state.files = []  # [{"name": str, "content": str, "type": str, "preview": str}]
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 4096  # По умолчанию максимум

# =========================
# Sidebar: управление
# =========================
with st.sidebar:
    st.header("⚙️ Настройки")
    current_model = st.selectbox(
        "Выбери модель",
        options=["sonoma", "deepseek"],
        format_func=lambda x: "Sonoma (Python Dev)" if x == "sonoma" else "DeepSeek (General AI)",
        index=0 if st.session_state.model == "sonoma" else 1
    )
    st.session_state.model = current_model

    # Новый слайдер для max_tokens
    current_max_tokens = st.slider(
        "📏 Макс. токенов для ответа (500–4096)",
        min_value=500,
        max_value=4096,
        value=st.session_state.max_tokens,
        help="Больше токенов = длиннее ответ, но дороже и медленнее. Учитывай input (история + файлы)."
    )
    st.session_state.max_tokens = current_max_tokens

    thinking_mode = st.checkbox("🧠 Режим обдумывания", value=False)

    # Загрузка файлов
    uploaded_files = st.file_uploader(
        "📁 Загрузить файлы", accept_multiple_files=True,
        type=['txt', 'pdf', 'jpg', 'png', 'csv', 'json', 'py'],
        help="Поддерживаемые типы: текст, PDF, изображения, CSV, JSON, Python. Макс. размер: 10MB на файл."
    )

    # Обработка загруженных файлов
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.warning(f"❌ Файл {uploaded_file.name} слишком большой (>10MB). Пропущен.")
                continue
            file_info = process_file(uploaded_file)
            if file_info:
                # Избегаем дубликатов по имени
                if not any(f['name'] == file_info['name'] for f in st.session_state.files):
                    st.session_state.files.append(file_info)
                    st.success(f"✅ Загружен: {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ Файл {uploaded_file.name} уже загружен.")

        # Показать список загруженных файлов
        if st.session_state.files:
            with st.expander("📋 Загруженные файлы", expanded=True):
                for file_info in st.session_state.files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{file_info['name']}** ({file_info['type']})")
                        if file_info['type'] == 'csv':
                            # Для CSV показываем мини-preview в sidebar
                            try:
                                df = pd.read_csv(io.StringIO(file_info['content']))
                                st.dataframe(df.head(5), use_container_width=True, height=150)
                            except:
                                st.markdown(file_info['preview'][:200] + "...")
                        else:
                            preview_text = file_info['preview'][:300] + "..." if len(file_info['preview']) > 300 else file_info['preview']
                            st.markdown(preview_text)
                    with col2:
                        if st.button("❌ Удалить", key=f"del_{file_info['name']}"):
                            st.session_state.files = [f for f in st.session_state.files if f['name'] != file_info['name']]
                            st.rerun()

    # Кнопки управления
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистить чат"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Очистить файлы"):
            st.session_state.files = []
            st.rerun()

    # Скачивание чата (всегда доступно, если есть сообщения)
    if st.session_state.messages:
        st.markdown("---")
        output = io.StringIO()
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            output.write(f"{role}: {msg['content']}\n\n")
        st.download_button(
            label="📤 Скачать чат (TXT)",
            data=output.getvalue(),
            file_name="chat.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("Нет сообщений для экспорта.")

# =========================
# Функция обработки файлов
# =========================
@st.cache_data
def process_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает загруженный файл и возвращает словарь с информацией.
    Возвращает None, если тип не поддерживается.
    """
    # Сохраняем оригинальный bytes для повторного использования
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Сброс курсора

    mime_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or ''
    extension = mimetypes.guess_extension(mime_type) or mimetypes.guess_extension(uploaded_file.name) or ''
    file_type = mime_type.split('/')[-1] if '/' in mime_type else extension[1:] if extension else 'unknown'

    content = ""  # Полное содержимое для API (текст)
    preview = f"[Файл: {uploaded_file.name}, тип: {file_type}, размер: {len(file_bytes)} байт]"

    try:
        if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            img = Image.open(io.BytesIO(file_bytes))
            st.image(img, caption=uploaded_file.name, width=200, use_column_width=False)
            content = f"Изображение {uploaded_file.name}: {img.size[0]}x{img.size[1]} пикселей. (Для анализа опиши, что на изображении, если нужно.)"
            preview = f"[Изображение: {uploaded_file.name}, {img.size[0]}x{img.size[1]}, {len(file_bytes)} байт]"

        elif extension == '.pdf':
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            content = text
            preview = text[:500] + "..." if len(text) > 500 else text
            preview = f"[PDF: {uploaded_file.name}]\n{preview}"

        elif extension in ['.txt', '.py']:
            content = file_bytes.decode("utf-8", errors="ignore")
            preview = content[:500] + "..." if len(content) > 500 else content
            preview = f"[Текст: {uploaded_file.name}]\n{preview}"

        elif extension == '.csv':
            content = file_bytes.decode("utf-8", errors="ignore")
            try:
                df = pd.read_csv(io.StringIO(content))
                preview = f"CSV с {len(df)} строками и колонками: {list(df.columns)}. Первые строки:\n{df.head(3).to_string()}"
            except:
                preview = content[:500] + "..."
            preview = f"[CSV: {uploaded_file.name}]\n{preview}"

        elif extension == '.json':
            content = file_bytes.decode("utf-8", errors="ignore")
            try:
                data = json.loads(content)
                preview = json.dumps(data, indent=2, ensure_ascii=False)[:500] + "..."
            except:
                preview = content[:500] + "..."
            preview = f"[JSON: {uploaded_file.name}]\n{preview}"

        else:
            content = file_bytes.decode("utf-8", errors="ignore")[:2000]  # Ограничение для неизвестных
            preview = content[:500] + "..."
            preview = f"[Файл: {uploaded_file.name} (неизвестный тип)]\n{preview}"

        return {
            "name": uploaded_file.name,
            "content": content,
            "type": file_type,
            "preview": preview
        }
    except Exception as e:
        st.error(f"Ошибка обработки {uploaded_file.name}: {e}")
        return None

# =========================
# Вывод чата
# =========================
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
if prompt := st.chat_input("Введите сообщение или запрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # =========================
    # Генерация ответа
    # =========================
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Обдумываю... (использую max_tokens=" + str(st.session_state.max_tokens) + ")")

        system_prompt = SYSTEM_PROMPTS[st.session_state.model].copy()
        if thinking_mode:
            system_prompt["content"] += "\nThink step by step and explain clearly."

        # Собираем контекст: системный + последние 20 сообщений + файлы (до 5 последних, чтобы не превысить токены)
        api_messages = [system_prompt]

        # Добавляем файлы как дополнительные user-сообщения (последние 5)
        recent_files = st.session_state.files[-5:]
        for file_info in recent_files:
            api_messages.append({
                "role": "user",
                "content": f"Вот содержимое файла {file_info['name']} (тип: {file_info['type']}):\n{file_info['content']}"
            })

        # Добавляем последние сообщения чата
        api_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-20:]
        ]

        try:
            model_name = "openrouter/sonoma-sky-alpha" if st.session_state.model == "sonoma" else "deepseek/deepseek-chat:v3.1"
            completion = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                temperature=0.7,
                max_tokens=st.session_state.max_tokens  # Динамически из сессии (максимум!)
            )
            reply = completion.choices[0].message.content
            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            error_msg = f"⚠️ Ошибка генерации: {str(e)}. Проверьте API-ключ, модель и лимит токенов (input может превышать контекст)."
            placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
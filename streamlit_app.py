import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import pandas as pd
import mimetypes
from typing import List, Dict, Any, Optional

# =========================
# Настройка страницы
# =========================
st.set_page_config(page_title="🤖 Мульти-Чат", page_icon="🤖", layout="wide")

st.title("🤖 Мульти-Чат: Sonoma & DeepSeek")
st.caption("Обращайся к любой из моделей. Русский/английский язык поддерживается.")

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
    st.session_state.files = []  # Список обработанных файлов: [{"name": str, "content": str, "type": str, "preview": str}]

# =========================
# Sidebar: управление
# =========================
with st.sidebar:
    st.header("⚙️ Настройки")
    st.session_state.model = st.selectbox(
        "Выбери модель",
        options=["sonoma", "deepseek"],
        format_func=lambda x: "Sonoma (Python Dev)" if x == "sonoma" else "DeepSeek (General AI)",
        index=0 if st.session_state.model == "sonoma" else 1
    )
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
                st.session_state.files.append(file_info)
                st.success(f"✅ Загружен: {uploaded_file.name}")

        # Показать список загруженных файлов
        if st.session_state.files:
            with st.expander("📋 Загруженные файлы", expanded=True):
                for file_info in st.session_state.files:
                    st.write(f"**{file_info['name']}** ({file_info['type']})")
                    st.markdown(file_info['preview'][:300] + "..." if len(file_info['preview']) > 300 else file_info['preview'])

    # Кнопки управления
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистить чат"):
            st.session_state.messages = []
            st.rerun()  # Используем st.rerun() вместо experimental_rerun() (Streamlit >=1.28)
    with col2:
        if st.button("🗑️ Очистить файлы"):
            st.session_state.files = []
            st.rerun()

    # Скачивание чата (всегда доступно, если есть сообщения)
    if st.session_state.messages:
        if st.button("📤 Скачать чат"):
            output = io.StringIO()
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                output.write(f"{role}: {msg['content']}\n\n")
            st.download_button(
                label="Скачать TXT",
                data=output.getvalue(),
                file_name="chat.txt",
                mime="text/plain"
            )
    else:
        st.info("Нет сообщений для экспорта.")

# =========================
# Функция обработки файлов
# =========================
@st.cache_data  # Кэшируем для производительности (если нужно)
def process_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает загруженный файл и возвращает словарь с информацией.
    Возвращает None, если тип не поддерживается.
    """
    # Сохраняем оригинальный bytes для повторного использования
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Сброс курсора, но поскольку мы сохранили bytes, используем их

    extension = mimetypes.guess_extension(uploaded_file.type) or ''
    file_type = uploaded_file.type or extension[1:]  # Убираем точку

    if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            st.image(img, caption=uploaded_file.name, width=200, use_column_width=False)
            preview = f"[Изображение: {uploaded_file.name}, размер {len(file_bytes)} байт]"
            return {
                "name": uploaded_file.name,
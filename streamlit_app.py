import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import pandas as pd
import mimetypes

# =========================
# Настройка страницы
# =========================
st.set_page_config(page_title="🤖 Мульти-Чат", page_icon="🤖", layout="wide")

st.title("🤖 Мульти-Чат: Sonoma & DeepSeek")
st.caption("Обращайся к любой из моделей. Русский/английский язык поддерживается.")

# =========================
# Секреты и API
# =========================
def get_client(api_key: str):
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

# =========================
# Sidebar: управление
# =========================
with st.sidebar:
    st.header("⚙️ Настройки")
    st.session_state.model = st.selectbox(
        "Выбери модель",
        options=["sonoma", "deepseek"],
        format_func=lambda x: "Sonoma (Python Dev)" if x == "sonoma" else "DeepSeek (General AI)"
    )
    thinking_mode = st.checkbox("🧠 Режим обдумывания", value=False)
    uploaded_files = st.file_uploader(
        "📁 Загрузить файлы", accept_multiple_files=True,
        type=['txt', 'pdf', 'jpg', 'png', 'csv', 'json', 'py']
    )
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = []
        st.experimental_rerun()
    if st.button("📤 Скачать чат"):
        if st.session_state.messages:
            output = io.StringIO()
            for msg in st.session_state.messages:
                output.write(f"{msg['role'].title()}: {msg['content']}\n\n")
            st.download_button("Скачать TXT", data=output.getvalue(), file_name="chat.txt")
        else:
            st.warning("Нет сообщений для экспорта.")

# =========================
# Функция обработки файлов
# =========================
def process_file(uploaded_file):
    extension = mimetypes.guess_extension(uploaded_file.type) or ''
    if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
        img = Image.open(uploaded_file)
        st.image(img, caption=uploaded_file.name, width=200)
        return f"[Image file: {uploaded_file.name}, size {uploaded_file.size} bytes]"
    elif extension == '.pdf':
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        preview = text[:5000] + "..." if len(text) > 5000 else text
        return f"[PDF file: {uploaded_file.name}]\n{preview}"
    elif extension in ['.txt', '.py', '.csv', '.json']:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        preview = content[:5000] + "..." if len(content) > 5000 else content
        return f"[File: {uploaded_file.name}]\n{preview}"
    else:
        return f"[File: {uploaded_file.name} — unsupported type]"

# =========================
# Добавление файлов в чат
# =========================
if uploaded_files:
    for f in uploaded_files:
        file_msg = process_file(f)
        st.session_state.messages.append({"role": "user", "content": file_msg})

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
        placeholder.markdown("⏳ Обдумываю...")

        system_prompt = SYSTEM_PROMPTS[st.session_state.model].copy()
        if thinking_mode:
            system_prompt["content"] += "\nThink step by step and explain clearly."

        # Собираем последние 20 сообщений
        api_messages = [system_prompt] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-20:]
        ]

        try:
            completion = client.chat.completions.create(
                model="openrouter/sonoma-sky-alpha" if st.session_state.model == "sonoma" else "deepseek/deepseek-chat-v3.1:free",
                messages=api_messages,
                temperature=0.7,
                max_tokens=20000
            )
            reply = completion.choices[0].message.content
            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            placeholder.markdown(f"⚠️ Ошибка: {e}")

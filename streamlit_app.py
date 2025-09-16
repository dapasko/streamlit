import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import mimetypes

# =========================
# Настройка API ключа
# =========================
api_key = st.secrets["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# =========================
# Системные промпты
# =========================
BASE_SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are Sonoma, an experienced fullstack Python developer built by Oak AI. 
Your users are Russian-speaking clients who may come to you with any request: from coding help (Python, Streamlit, web dev, APIs, databases) to general advice, file analysis, or everyday questions. 
You know or can figure out answers to everything — analyze problems logically, provide code snippets, step-by-step guides, and practical solutions. 
Always respond in Russian if the query is in Russian; otherwise, use English. Be friendly, helpful, witty, and use emojis where appropriate. 
Structure responses: use ```python for code blocks, bullet points or numbered lists for steps, bold for key points. 
If a file is uploaded, analyze it as a developer: check for errors in code, suggest improvements, or summarize content. 
For images, since you are text-based, ask the user to describe it or suggest Python code to process it (e.g., with Pillow). 
Do not generate harmful content or role-play unless asked. You have session memory but no persistent storage."""
}

THINKING_MODE_PROMPT = "Before answering, think step by step: 1) Analyze the problem or request. 2) Outline the key steps or considerations. 3) Provide a clear, detailed solution or code. This ensures thorough and accurate responses."

# =========================
# Настройка страницы
# =========================
st.set_page_config(page_title="🤖 Sonoma — Fullstack Python Dev", page_icon="🤖", layout="wide")
st.title("🤖 Sonoma от Oak AI: Твой Fullstack Python Разработчик")
st.markdown("---")
st.caption("Привет! Я Sonoma, опытный fullstack Python разработчик. Обращайся с любой просьбой — от кода и анализа файлов до советов. Русскоязычные клиенты? Отвечаю на русском! 😊")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Управление")
    thinking_mode = st.checkbox("🧠 Режим максимального обдумывания", value=False)
    st.info("🛠️ Модель: openrouter/sonoma-sky-alpha (текстовая, без vision)")

    uploaded_files = st.file_uploader(
        "📁 Загрузи файл(ы) для анализа",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'jpg', 'png', 'csv', 'json', 'py']
    )

    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = [BASE_SYSTEM_PROMPT]
        st.rerun()

    if st.button("📤 Экспорт чата"):
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            output = io.StringIO()
            for msg in st.session_state.messages[1:]:
                if isinstance(msg["content"], str):
                    content_str = msg["content"]
                else:  # Файловые сообщения
                    if "file_type" in msg:
                        content_str = f"[Файл: {msg['name']} ({msg['file_type']})] {msg.get('preview', '')}"
                    else:
                        content_str = str(msg["content"])[:200] + "..."
                output.write(f"{msg['role'].title()}: {content_str}\n\n")
            st.download_button(
                label="Скачать как TXT",
                data=output.getvalue(),
                file_name="sonoma_chat.txt",
                mime="text/plain"
            )
        else:
            st.warning("Нет сообщений для экспорта.")

# =========================
# Инициализация сессии
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [BASE_SYSTEM_PROMPT]

current_system_prompt = BASE_SYSTEM_PROMPT.copy()
if thinking_mode:
    current_system_prompt["content"] += "\n\n" + THINKING_MODE_PROMPT

# =========================
# Обработка загруженных файлов
# =========================
if uploaded_files:
    st.write("**Загруженные файлы для анализа:**")
    for uploaded_file in uploaded_files:
        file_extension = mimetypes.guess_extension(uploaded_file.type) or ''
        preview_text = ""
        file_type = "text"

        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, width=200)
            preview_text = f"Изображение '{uploaded_file.name}' (размер: {uploaded_file.size} байт). Опиши его, и я помогу с Python-кодом для обработки."
            file_type = "image"
            content_parts = [{"type": "text", "text": preview_text}]

        elif file_extension.lower() == '.pdf':
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            preview_text = text[:1000] + "... (полный текст в промпте)"
            file_type = "pdf"
            content_parts = [{"type": "text", "text": f"PDF '{uploaded_file.name}':\n\n{text[:4000]}..."}]

        elif file_extension.lower() in ['.txt', '.csv', '.json', '.md', '.py']:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            preview_text = text[:500] + "... (полный текст в промпте)"
            file_type = "text"
            content_parts = [{"type": "text", "text": f"Файл '{uploaded_file.name}':\n\n{text[:4000]}..."}]

        else:
            preview_text = f"Файл '{uploaded_file.name}' не полностью поддерживается. Опиши, что в нём."
            file_type = "other"
            content_parts = [{"type": "text", "text": preview_text}]

        st.write(f"📄 {uploaded_file.name} ({file_type})")
        st.write(preview_text)

        st.session_state.messages.append({
            "role": "user",
            "content": content_parts,
            "file_type": file_type,
            "name": uploaded_file.name,
            "preview": preview_text
        })

# =========================
# Отображение истории сообщений
# =========================
for msg in st.session_state.messages[1:]:
    role = msg["role"]
    if role not in ["user", "assistant"]:
        continue
    content = msg["content"]
    if isinstance(content, str):
        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

# =========================
# Ввод нового сообщения
# =========================
if prompt := st.chat_input("Введите сообщение или запрос... (например, 'Проанализируй код в файле')"):
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерация ответа
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Обдумываю и генерирую ответ...")

        try:
            api_messages = [current_system_prompt]
            for msg in st.session_state.messages[1:]:
                if isinstance(msg["content"], str):
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                else:
                    text_content = " ".join([part["text"] for part in msg["content"] if part["type"] == "text"])
                    api_messages.append({"role": msg["role"], "content": text_content})

            if len(api_messages) > 21:
                api_messages = [current_system_prompt] + api_messages[-20:]

            temperature = 0.3 if thinking_mode else 0.7
            completion = client.chat.completions.create(
                model="openrouter/sonoma-sky-alpha",
                messages=api_messages,
                temperature=temperature,
                max_tokens=3000,
            )

            reply = completion.choices[0].message.content
            message_placeholder.markdown(reply, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)}. Проверь API-ключ или попробуй упростить запрос."
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =========================
# Отладка (опционально)
# =========================
if st.checkbox("📊 Статистика сессии"):
    st.write(f"Режим обдумывания: {'Включён' if thinking_mode else 'Выключен'}")
    st.write(f"Сообщений: {len(st.session_state.messages)}")

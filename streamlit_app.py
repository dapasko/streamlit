import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader  # Для PDF
from PIL import Image  # Для превью изображений
import mimetypes  # Для типов файлов
import os  # Для метаданных файлов

# Достаём ключ из секретов Streamlit Cloud
api_key = st.secrets["OPENROUTER_API_KEY"]

# Создаём клиента
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Базовый системный промпт для Sonoma как fullstack Python разработчика
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

# Динамический промпт для режима обдумывания
THINKING_MODE_PROMPT = "Before answering, think step by step: 1) Analyze the problem or request. 2) Outline the key steps or considerations. 3) Provide a clear, detailed solution or code. This ensures thorough and accurate responses."

st.set_page_config(page_title="🤖 Sonoma — Fullstack Python Dev", page_icon="🤖", layout="wide")

st.title("🤖 Sonoma от Oak AI: Твой Fullstack Python Разработчик")
st.markdown("---")
st.caption("Привет! Я Sonoma, опытный fullstack Python разработчик. Обращайся с любой просьбой — от кода и анализа файлов до советов. Русскоязычные клиенты? Отвечаю на русском! 😊")

# Sidebar для управления
with st.sidebar:
    st.header("Управление")

    # Режим максимального обдумывания
    thinking_mode = st.checkbox("🧠 Режим максимального обдумывания", value=False)

    # Модель фиксирована: только Sonoma
    st.info("🛠️ Модель: openrouter/sonoma-sky-alpha (текстовая, без vision)")

    # Загрузка файлов
    uploaded_files = st.file_uploader("📁 Загрузи файл(ы) для анализа", accept_multiple_files=True, type=['txt', 'pdf', 'jpg', 'png', 'csv', 'json', 'py'])

    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = [BASE_SYSTEM_PROMPT]
        st.rerun()

    if st.button("📤 Экспорт чата"):
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            output = io.StringIO()
            for msg in st.session_state.messages[1:]:
                if isinstance(msg["content"], str):
                    content_str = msg["content"]
                else:  # Файл: упрощённо
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

# Инициализация сессии с системным промптом
if "messages" not in st.session_state:
    st.session_state.messages = [BASE_SYSTEM_PROMPT]

# Динамический системный промпт с режимом обдумывания
current_system_prompt = BASE_SYSTEM_PROMPT.copy()
if thinking_mode:
    current_system_prompt["content"] += "\n\n" + THINKING_MODE_PROMPT

# Обработка загруженных файлов (добавляем в чат)
if uploaded_files:
    with st.chat_message("user", avatar_icon="👤"):
        st.write("**Загруженные файлы для анализа:**")
        file_messages = []
        for uploaded_file in uploaded_files:
            file_extension = mimetypes.guess_extension(uploaded_file.type) or ''
            preview_text = ""
            content_parts = []
            file_type = "text"

            if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # Изображение: превью + предупреждение
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, width=200)
                preview_text = f"Изображение '{uploaded_file.name}' (размер: {uploaded_file.size} байт). Опиши его, и я помогу с Python-кодом для обработки (Pillow, OpenCV)."
                content_parts = [{"type": "text", "text": preview_text}]
                file_type = "image"

            elif file_extension.lower() == '.pdf':
                # PDF: извлекаем текст
                reader = PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                preview_text = text[:1000] + "... (полный текст в промпте)"
                content_parts = [{"type": "text", "text": f"PDF '{uploaded_file.name}':\n\n{text[:4000]}..."}]
                file_type = "pdf"

            elif file_extension.lower() in ['.txt', '.csv', '.json', '.md', '.py']:
                # Текст: читаем
                text = uploaded_file.read().decode("utf-8", errors="ignore")
                preview_text = text[:500] + "... (полный текст в промпте)"
                content_parts = [{"type": "text", "text": f"Файл '{uploaded_file.name}':\n\n{text[:4000]}..."}]
                file_type = "text"

            else:
                preview_text = f"Файл '{uploaded_file.name}' не полностью поддерживается. Опиши, что в нём."
                content_parts = [{"type": "text", "text": preview_text}]
                file_type = "other"

            st.write(f"📄 {uploaded_file.name} ({file_type})")
            st.write(preview_text)

            # Сохраняем в сессию
            file_msg = {
                "role": "user",
                "content": content_parts,
                "file_type": file_type,
                "name": uploaded_file.name,
                "preview": preview_text
            }
            st.session_state.messages.append(file_msg)
            file_messages.append(file_msg)

        # Показ в чате (если нужно, но превью выше)

# Показ истории сообщений (пропускаем системный)
for msg in st.session_state.messages[1:]:
    if isinstance(msg["content"], str):  # Обычное сообщение
        with st.chat_message(msg["role"], avatar_icon=("👤" if msg["role"] == "user" else "🤖")):
            st.markdown(msg["content"], unsafe_allow_html=True)
    # Файловые: UI показан при загрузке, не дублируем

# Ввод текстового сообщения
if prompt := st.chat_input("Введите сообщение или запрос... (например, 'Проанализируй код в файле')"):
    # Добавляем сообщение пользователя
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user", avatar_icon="👤"):
        st.markdown(prompt)

    # Генерация ответа
    with st.chat_message("assistant", avatar_icon="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Обдумываю и генерирую ответ...")

        try:
            # Формируем API-сообщения
            api_messages = [current_system_prompt]
            for msg in st.session_state.messages[1:]:
                if isinstance(msg["content"], str):
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                else:  # Файл: flatten текст
                    text_content = " ".join([part["text"] for part in msg["content"] if part["type"] == "text"])
                    api_messages.append({"role": msg["role"], "content": text_content})

            # Ограничение длины (последние 20 сообщений)
            if len(api_messages) > 21:
                api_messages = [current_system_prompt] + api_messages[-20:]

            # API-запрос: temperature зависит от режима
            temperature = 0.3 if thinking_mode else 0.7
            completion = client.chat.completions.create(
                model="openrouter/sonoma-sky-alpha",
                messages=api_messages,
                temperature=temperature,
                max_tokens=3000,  # Увеличил для детальных ответов
            )

            reply = completion.choices[0].message.content
            message_placeholder.markdown(reply, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)}. Проверь API-ключ или попробуй упростить запрос."
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Отладка (опционально)
if st.checkbox("📊 Статистика сессии"):
    st.write(f"Режим обдумывания: {'Включён' if thinking_mode else 'Выключен'}")
    st.write(f"Сообщений: {len(st.session_state.messages)}")
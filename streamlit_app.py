import streamlit as st
from openai import OpenAI
import io
from PyPDF2 import PdfReader
from PIL import Image
import mimetypes
import pandas as pd  # Для обработки CSV/JSON
import logging

# =========================
# Constants и Logging
# =========================
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_HISTORY = 20  # Максимум сообщений для API (с учётом summarization)
SUPPORTED_TYPES = ['.txt', '.pdf', '.jpg', '.png', '.csv', '.json', '.py']

# Logging для отладки (не выводит в UI, но полезно)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Настройка API клиента
# =========================
@st.cache_resource
def get_client():
    """Инициализация клиента OpenAI с проверкой ключа."""
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("❌ API-ключ не найден в secrets.toml. Добавьте OPENROUTER_API_KEY = 'your_key'.")
        st.stop()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client

client = get_client()

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
If a file is uploaded, analyze it as developer: check for errors in code, suggest improvements, or summarize content. 
For images, since you are text-based, ask the user to describe it or suggest Python code to process it (e.g., with Pillow). 
Do not generate harmful content or role-play unless asked. You have session memory but no persistent storage."""
}

THINKING_MODE_PROMPT = "Before answering, think step by step: 1) Analyze the problem or request. 2) Outline the key steps or considerations. 3) Provide a clear, detailed solution or code. This ensures thorough and accurate responses."

# =========================
# Функции для обработки файлов (улучшенная с кэшем и валидацией)
# =========================
@st.cache_data
def process_file(uploaded_file):
    """Обработка одного файла: валидация, парсинг, preview."""
    if uploaded_file.size > MAX_FILE_SIZE:
        return {"error": f"Файл {uploaded_file.name} слишком большой ({uploaded_file.size / 1024 / 1024:.1f} MB). Максимум: 10MB."}

    file_extension = mimetypes.guess_extension(uploaded_file.type) or ''
    if file_extension not in SUPPORTED_TYPES:
        return {"error": f"Тип файла {file_extension} не поддерживается. Поддержка: {', '.join(SUPPORTED_TYPES)}"}

    uploaded_file.seek(0)  # Сброс позиции в потоке
    progress_bar = st.progress(0)

    try:
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, width=200)
            preview = f"Изображение '{uploaded_file.name}' (размер: {uploaded_file.size} байт). Опиши его, и я помогу с Python-кодом для обработки (например, Pillow)."
            content = preview  # Для промпта
            file_type = "image"
            progress_bar.progress(1)

        elif file_extension.lower() == '.pdf':
            pdf_bytes = uploaded_file.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text += page.extract_text() + "\n"
                progress_bar.progress((i + 1) / num_pages if num_pages > 0 else 1)
            preview = text[:1000] + "..." if len(text) > 1000 else text
            content = text[:4000] + "..." if len(text) > 4000 else text
            file_type = "pdf"

        elif file_extension.lower() in ['.csv']:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)  # Preview таблицы
            preview = df.head(5).to_string(index=False) + "..." if len(df) > 5 else df.to_string(index=False)
            content = df.to_json(orient='records')  # Полный JSON для промпта
            file_type = "csv"
            progress_bar.progress(1)

        elif file_extension.lower() == '.json':
            json_str = uploaded_file.read().decode("utf-8", errors="ignore")
            json_data = pd.read_json(io.StringIO(json_str))
            st.json(json_data.head().to_dict())  # Preview JSON
            preview = str(json_data.head().to_dict())[:500] + "..."
            content = json_str[:4000] + "..." if len(json_str) > 4000 else json_str
            file_type = "json"
            progress_bar.progress(1)

        else:  # .txt, .py, .md и т.д.
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            preview = text[:500] + "..." if len(text) > 500 else text
            content = text[:4000] + "..." if len(text) > 4000 else text
            file_type = "text"
            progress_bar.progress(1)

        return {"type": file_type, "preview": preview, "content": content, "no_error": True}

    except Exception as e:
        logger.error(f"Ошибка обработки файла {uploaded_file.name}: {e}")
        return {"error": f"Ошибка обработки: {str(e)}"}

# =========================
# Функция для сборки сообщений (с summarization для длинной истории)
# =========================
def build_messages(system_prompt, session_messages):
    """Сборка сообщений для API. Если история длинная — summarization старой части."""
    api_messages = [system_prompt]

    if len(session_messages) > MAX_HISTORY + 1:
        # Разделяем на старую и новую историю
        old_limit = MAX_HISTORY // 2
        old_msgs = []
        for msg in session_messages[1:1 + old_limit]:
            if msg["role"] in ["user", "assistant"]:
                if isinstance(msg["content"], str):
                    old_msgs.append({"role": msg["role"], "content": msg["content"]})
                else:  # Файловые сообщения
                    text_content = " ".join([part["text"] for part in msg["content"] if isinstance(part, dict) and part.get("type") == "text"])
                    old_msgs.append({"role": msg["role"], "content": text_content})

        try:
            # Summarization старой истории
            summary_prompt = {"role": "system", "content": "Суммаризуй предыдущую историю чата кратко (1-2 абзаца), сохраняя ключевые моменты для контекста."}
            model = st.session_state.get("model", "openrouter/sonoma-sky-alpha")
            summary_completion = client.chat.completions.create(
                model=model,
                messages=[summary_prompt] + old_msgs,
                max_tokens=500,
                temperature=0.3
            )
            summary = summary_completion
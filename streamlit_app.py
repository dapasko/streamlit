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
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("❌ API-ключ не найден в secrets. Добавьте OPENROUTER_API_KEY.")
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
                progress_bar.progress((i + 1) / num_pages)
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
            json_data = pd.read_json(uploaded_file)
            st.json(json_data.head().to_dict())  # Preview JSON
            preview = str(json_data.head().to_dict())[:500] + "..."
            content = uploaded_file.getvalue().decode("utf-8")
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
        old_msgs = [{"role": msg["role"], "content": msg["content"] if isinstance(msg["content"], str) else " ".join([part["text"] for part in msg["content"]])} for msg in session_messages[1:1 + old_limit]]

        try:
            # Summarization старой истории
            summary_prompt = {"role": "system", "content": "Суммаризуй предыдущую историю чата кратко (1-2 абзаца), сохраняя ключевые моменты для контекста."}
            summary_completion = client.chat.completions.create(
                model=st.session_state.get("model", "openrouter/sonoma-sky-alpha"),
                messages=[summary_prompt] + old_msgs,
                max_tokens=500,
                temperature=0.3
            )
            summary = summary_completion.choices[0].message.content
            api_messages.append({"role": "system", "content": f"Предыдущая история (краткий обзор): {summary}"})

            # Добавляем только недавние
            recent_msgs = session_messages[1 + old_limit:]
        except Exception as e:
            logger.warning(f"Ошибка summarization: {e}. Используем простую обрезку.")
            recent_msgs = session_messages[-(MAX_HISTORY // 2):]
    else:
        recent_msgs = session_messages

    # Добавляем недавние сообщения
    for msg in recent_msgs:
        if msg["role"] in ["user", "assistant"]:
            if isinstance(msg["content"], str):
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            else:  # Файловые сообщения
                text_content = " ".join([part["text"] for part in msg["content"] if isinstance(part, dict) and part.get("type") == "text"])
                api_messages.append({"role": msg["role"], "content": text_content})

    return api_messages

# =========================
# Функция генерации ответа
# =========================
def generate_response(api_messages, model, temperature):
    """Генерация ответа с обработкой ошибок."""
    try:
        with st.spinner("⏳ Генерирую ответ..."):
            completion = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=3000,
            )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"API ошибка: {e}")
        if "rate limit" in str(e).lower():
            return "⚠️ Достигнут лимит запросов. Подождите или проверьте API-ключ."
        elif "invalid" in str(e).lower():
            return "⚠️ Неверный API-ключ или модель. Проверьте настройки."
        else:
            return f"⚠️ Ошибка API: {str(e)}. Попробуйте упростить запрос."

# =========================
# Функция экспорта чата
# =========================
def export_chat(messages):
    """Экспорт истории в TXT."""
    if len(messages) <= 1:
        st.warning("Нет сообщений для экспорта.")
        return

    output = io.StringIO()
    for msg in messages[1:]:
        if msg["role"] not in ["user", "assistant"]:
            continue
        if isinstance(msg["content"], str):
            content_str = msg["content"]
        else:
            if "preview" in msg:
                content_str = f"[Файл: {msg['name']} ({msg.get('file_type', 'unknown')})] {msg['preview']}"
            else:
                content_str = str(msg["content"])[:200] + "..."
        output.write(f"{msg['role'].title()}: {content_str}\n\n")

    st.download_button(
        label="📥 Скачать чат как TXT",
        data=output.getvalue(),
        file_name="sonoma_chat.txt",
        mime="text/plain"
    )

# =========================
# Настройка страницы
# =========================
st.set_page_config(
    page_title="🤖 Sonoma — Fullstack Python Dev",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Sonoma от Oak AI: Твой Fullstack Python Разработчик")
st.markdown("---")
st.caption("Привет! Я Sonoma, опытный fullstack Python разработчик. Обращайся с любой просьбой — от кода и анализа файлов до советов. Русскоязычные клиенты? Отвечаю на русском! 😊")

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Управление")
    thinking_mode = st.checkbox("🧠 Режим максимального обдумывания", value=False)
    st.session_state.model = st.selectbox("🤖 Модель", ["openrouter/sonoma-sky-alpha"], key="model_select")  # Расширяемо
    temperature = st.slider("🌡️ Температура (креативность)", 0.0, 1.0, 0.7 if not thinking_mode else 0.3)

    st.subheader
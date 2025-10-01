import streamlit as st
import requests
import os, json, time
from pathlib import Path

# ======================
# Конфиги
# ======================
MODEL = "x-ai/grok-4-fast:free"
API_KEY = st.secrets["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

BATCH_SIZE = 50
HISTORY_FILE = Path("translation_history.json")


# ======================
# Функции
# ======================
def load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return {}


def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def call_model(batch_lines):
    """Отправка батча в модель"""
    messages = [
        {"role": "system", "content": "Ты переводчик локализаций для Crusader Kings 3. "
                                      "Переводи только текст в кавычках, сохраняя формат YAML. "
                                      "Не меняй ключи, не убирай :0, только перевод внутри кавычек."},
        {"role": "user", "content": "\n".join(batch_lines)}
    ]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "CK3 Mod Translator"
    }

    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4000
    }

    for attempt in range(3):
        try:
            r = requests.post(API_URL, headers=headers, json=data, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return content.splitlines()
        except Exception as e:
            time.sleep(2 * (attempt+1))
            if attempt == 2:
                raise e
    return []


def translate_file(filename, content):
    """Основная функция перевода файла"""
    lines = content.splitlines()
    history = load_history()

    # ключ истории по имени файла
    file_key = filename
    if file_key not in history:
        history[file_key] = {"done_batches": 0, "translated": []}

    start_batch = history[file_key]["done_batches"]
    total_batches = (len(lines) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(start_batch, total_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx+1)*BATCH_SIZE, len(lines))
        batch_lines = lines[start:end]

        st.info(f"Переводим батч {batch_idx+1}/{total_batches}...")
        translated_batch = call_model(batch_lines)

        # если модель вернула меньше строк — дополним
        if len(translated_batch) < len(batch_lines):
            translated_batch += batch_lines[len(translated_batch):]

        history[file_key]["translated"].extend(translated_batch)
        history[file_key]["done_batches"] = batch_idx+1
        save_history(history)

    return "\n".join(history[file_key]["translated"])


# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="CK3 Translator", layout="wide")
st.title("🎮 Crusader Kings III Translator (с батчами + историей)")

uploaded = st.file_uploader("Загрузите .yml файл локализации", type=["yml"])

if uploaded and st.button("Перевести"):
    text = uploaded.read().decode("utf-8", errors="replace")
    st.info("⏳ Начинаем перевод...")

    translated = translate_file(uploaded.name, text)

    st.success("✅ Перевод готов!")

    out_name = "l_russian_" + Path(uploaded.name).name
    st.download_button(
        label="📥 Скачать переведённый файл",
        data=translated.encode("utf-8"),
        file_name=out_name,
        mime="text/plain"
    )

    st.info(f"История сохранена в {HISTORY_FILE}")

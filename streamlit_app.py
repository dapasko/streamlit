import streamlit as st
import requests
import re
import time
from pathlib import Path

# ======================
# Конфиги
# ======================
MODEL = "x-ai/grok-4-fast:free"
API_KEY = st.secrets["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

BATCH_SIZE = 50

# История переводов в памяти
history = {}


# ======================
# Пост-обработка перевода
# ======================
def clean_translation(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # Удаляем дубли
        if line.strip() in cleaned:
            continue

        # Подмены терминов
        line = line.replace("автоceph", "автокеф")
        line = line.replace("Confirm", "Подтвердить")
        line = line.replace("Epic.", "Великолепно.")
        line = re.sub(r"\bрвение\b", "фервор", line)

        cleaned.append(line)

    # Проверка на английский текст
    for l in cleaned:
        if re.search(r"[A-Za-z]", l) and not l.startswith("l_russian"):
            print("⚠️ Остался английский:", l)

    return "\n".join(cleaned)


# ======================
# API вызов
# ======================
def call_model(batch_lines):
    """Отправка батча в модель"""
    messages = [
        {"role": "system", "content": "Ты переводчик локализаций для Crusader Kings 3. "
                                      "Переводи только текст в кавычках, сохраняя формат YAML. "
                                      "Не меняй ключи, не убирай :0, только перевод внутри кавычек. "
                                      "Используй историко-православную лексику: автокефалия, патриархат, ересь, фервор. "
                                      "Confirm → Подтвердить, Epic. → Великолепно."},
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
            print(f"📤 Отправляем батч ({len(batch_lines)} строк), попытка {attempt+1}")
            r = requests.post(API_URL, headers=headers, json=data, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            print("✅ Батч переведён")
            return content.splitlines()
        except Exception as e:
            print("⚠️ Ошибка:", e)
            time.sleep(2 * (attempt+1))
            if attempt == 2:
                raise e
    return []


# ======================
# Основная функция перевода
# ======================
def translate_file(filename, content):
    lines = content.splitlines()

    # Ключ истории по имени файла
    file_key = filename
    if file_key not in history:
        history[file_key] = {"done_batches": 0, "translated": []}

    start_batch = history[file_key]["done_batches"]
    total_batches = (len(lines) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(start_batch, total_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(lines))
        batch_lines = lines[start:end]

        print(f"➡️ Переводим батч {batch_idx+1}/{total_batches} ({start}-{end})")
        translated_batch = call_model(batch_lines)

        # если модель вернула меньше строк — дополним
        if len(translated_batch) < len(batch_lines):
            translated_batch += batch_lines[len(translated_batch):]

        history[file_key]["translated"].extend(translated_batch)
        history[file_key]["done_batches"] = batch_idx + 1

    final_text = "\n".join(history[file_key]["translated"])
    final_text = clean_translation(final_text)
    return final_text


# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="CK3 Translator", layout="wide")
st.title("🎮 Crusader Kings III Translator (батчи + пост-обработка)")

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

    st.info("ℹ️ Логи процесса смотрите в консоли (терминале)")

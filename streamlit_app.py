import streamlit as st
import requests
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================
# Конфиги
# ======================
MODEL = "x-ai/grok-4-fast:free"
API_KEY = st.secrets["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

BATCH_SIZE = 100      # размер батча (чем больше, тем меньше запросов)
MAX_WORKERS = 3       # сколько батчей обрабатываем параллельно

# История переводов в памяти
history = {}


# ======================
# Пост-обработка перевода
# ======================
def clean_translation(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if line.strip() in cleaned:
            continue

        # Подмены терминов
        line = line.replace("автоceph", "автокеф")
        line = line.replace("Confirm", "Подтвердить")
        line = line.replace("Epic.", "Великолепно.")
        line = re.sub(r"\bрвение\b", "фервор", line)

        cleaned.append(line)

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
        "max_tokens": 20000
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
    file_key = filename
    if file_key not in history:
        history[file_key] = {"done_batches": 0, "translated": []}

    total_batches = (len(lines) + BATCH_SIZE - 1) // BATCH_SIZE
    st.info(f"Всего батчей: {total_batches}")

    progress = st.progress(0)
    log_area = st.empty()

    batches = []
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(lines))
        batches.append((batch_idx, lines[start:end]))

    results = [None] * len(batches)

    # параллельно обрабатываем несколько батчей
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(call_model, b[1]): b[0] for b in batches}
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                translated_batch = future.result()
                results[idx] = translated_batch
                log_area.text(f"✅ Батч {idx+1}/{total_batches} готов")
            except Exception as e:
                log_area.text(f"⚠️ Ошибка в батче {idx+1}: {e}")
                results[idx] = batches[idx][1]  # если ошибка, оставляем оригинал

            done_count += 1
            progress.progress(done_count / total_batches)

    # собираем перевод обратно
    translated = []
    for batch_lines in results:
        translated.extend(batch_lines)

    final_text = "\n".join(translated)
    final_text = clean_translation(final_text)
    return final_text


# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="CK3 Translator", layout="wide")
st.title("🎮 Crusader Kings III Translator (батчи + параллель + пост-обработка)")

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

    st.info("ℹ️ Логи процесса смотрите в консоли и ниже во время работы")

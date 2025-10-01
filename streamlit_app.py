import streamlit as st
import requests
import re
import json
import time
from pathlib import Path

# ===============================
# Конфиги
# ===============================
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "x-ai/grok-4-fast:free"
API_KEY = st.secrets["OPENROUTER_API_KEY"]

LOC_LINE_RE = re.compile(r'^(\S+):\d+\s+"(.*)"\s*$')
MAX_RETRIES = 3


# ===============================
# Функции
# ===============================
def extract_entries(lines):
    """Ищет KEY:0 "Text" строки"""
    entries = []
    for i, ln in enumerate(lines):
        m = LOC_LINE_RE.match(ln)
        if m:
            entries.append({"idx": i, "key": m.group(1), "text": m.group(2)})
    return entries


def build_system_prompt():
    return (
        "Translate the given Crusader Kings 3 localisation lines from English to Russian. "
        "Preserve placeholders like [GetName], $VAR$, #TAG#, {foo}, <icon>. "
        "Return ONLY JSON array: [{\"original\":\"...\",\"translation\":\"...\"}, ...]"
    )


def call_model_batch(texts):
    system_prompt = build_system_prompt()
    user_prompt = "Translate these strings:\n" + json.dumps(texts, ensure_ascii=False)

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.post(OPENROUTER_API, headers=headers, json=body, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            # ищем массив JSON
            start = content.find("[")
            parsed = json.loads(content[start:])
            return [p["translation"] for p in parsed]
        except Exception as e:
            st.warning(f"Ошибка вызова API (попытка {attempt}): {e}")
            time.sleep(2 * attempt)
    return []


def translate_file(filename, content, batch_size=20):
    lines = content.splitlines()
    entries = extract_entries(lines)

    for start in range(0, len(entries), batch_size):
        batch = entries[start:start+batch_size]
        texts = [e["text"] for e in batch]
        translations = call_model_batch(texts)

        for e, tr in zip(batch, translations):
            safe_tr = tr.replace('"', '\\"')
            lines[e["idx"]] = f'{e["key"]}:0 "{safe_tr}"'

    return "\n".join(lines)


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="CK3 Translator", layout="wide")
st.title("🛠️ Crusader Kings III Translator")

uploaded = st.file_uploader("Загрузите .yml файл локализации", type=["yml"])
batch_size = st.slider("Размер батча (строк за один запрос)", 5, 50, 20, 5)

if uploaded and st.button("Перевести"):
    text = uploaded.read().decode("utf-8", errors="replace")
    st.info("⏳ Переводим...")
    translated = translate_file(uploaded.name, text, batch_size)
    st.success("✅ Перевод готов!")

    out_name = "l_russian_" + Path(uploaded.name).name
    st.download_button(
        label="📥 Скачать переведённый файл",
        data=translated.encode("utf-8"),
        file_name=out_name,
        mime="text/plain"
    )

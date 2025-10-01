import streamlit as st
import requests
import os
import io
import zipfile
import re
import json
import time
from pathlib import Path
from typing import List, Dict

# ===============================
# Конфиги
# ===============================
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "x-ai/grok-4-fast:free"
API_KEY = st.secrets["OPENROUTER_API_KEY"]

LOC_LINE_RE = re.compile(r'^(\S+):\d+\s+"(.*)"\s*$')
PLACEHOLDER_PATTERNS = [
    r'\[.*?\]', r'\{.*?\}', r'\#.*?\#', r'\$[A-Za-z0-9_]+\$', r'<.*?>'
]
PLACEHOLDER_RE = re.compile('|'.join(PLACEHOLDER_PATTERNS))

MAX_RETRIES = 3


# ===============================
# Вспомогательные функции
# ===============================
def parse_loc_file(content: str) -> List[str]:
    """Разбивает файл на строки"""
    return content.splitlines()


def extract_entries(lines: List[str]) -> List[Dict]:
    """Ищет KEY:0 "Text" строки"""
    entries = []
    for i, ln in enumerate(lines):
        m = LOC_LINE_RE.match(ln)
        if m:
            entries.append({"idx": i, "key": m.group(1), "text": m.group(2)})
    return entries


def build_system_prompt():
    return (
        "You are a professional translator. Translate the given English localisation strings for "
        "the Paradox/Crusader Kings 3 mod into Russian. Preserve all placeholders unchanged "
        "(examples: [GetRoot.GetName], #TAG#, $VARIABLE$, {foo}, <icon>). "
        "Return the translations as JSON array with keys 'original' and 'translation'."
    )


def call_model_batch(texts: List[str]) -> List[str]:
    system_prompt = build_system_prompt()
    user_prompt = "Translate these strings to Russian. Respond ONLY with JSON.\n" + json.dumps(texts, ensure_ascii=False)

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 20000
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "CK3-Translator-Streamlit"
    }

    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.post(OPENROUTER_API, headers=headers, json=body, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content[content.find("["):])  # если есть лишний текст
            return [p["translation"] for p in parsed]
        except Exception as e:
            time.sleep(2 * attempt)
            if attempt == MAX_RETRIES:
                raise e
    return []


def translate_file(name: str, content: str, batch_size: int = 15) -> str:
    lines = parse_loc_file(content)
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
st.title("🛠️ Crusader Kings III Translator (xAI Grok)")

st.markdown("Загрузите файлы локализации (.yml) или архив (.zip). Скрипт переведет их на русский и предложит скачать результат.")

uploaded = st.file_uploader("Выберите .yml или .zip", type=["yml", "zip"], accept_multiple_files=True)
batch_size = st.slider("Размер батча", 5, 50, 15, 5)

if uploaded and st.button("Перевести"):
    output_buffer = io.BytesIO()
    with zipfile.ZipFile(output_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in uploaded:
            filename = file.name
            data = file.read()

            if filename.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as modzip:
                    for inner_name in modzip.namelist():
                        if inner_name.endswith(".yml"):
                            text = modzip.read(inner_name).decode("utf-8", errors="replace")
                            st.info(f"Переводим {inner_name} ...")
                            translated = translate_file(inner_name, text, batch_size)
                            out_name = "l_russian_" + Path(inner_name).name
                            zf.writestr(out_name, translated)
            else:
                text = data.decode("utf-8", errors="replace")
                st.info(f"Переводим {filename} ...")
                translated = translate_file(filename, text, batch_size)
                out_name = "l_russian_" + Path(filename).name
                zf.writestr(out_name, translated)

    st.success("Перевод завершен!")
    st.download_button(
        "📥 Скачать переводы (zip)",
        data=output_buffer.getvalue(),
        file_name="ck3_translated.zip",
        mime="application/zip"
    )

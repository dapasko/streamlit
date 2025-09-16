import streamlit as st
from openai import OpenAI

# Достаём ключ из секретов Streamlit Cloud
api_key = st.secrets["OPENROUTER_API_KEY"]

# Создаём клиента
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

st.title("🤖 Чат с OpenRouter")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Показ истории
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Введите сообщение..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Генерация...")

        try:
            completion = client.chat.completions.create(
                model="openrouter/sonoma-sky-alpha",
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state.messages],
            )
            reply = completion.choices[0].message.content
            message_placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            message_placeholder.markdown(f"⚠️ Ошибка: {e}")

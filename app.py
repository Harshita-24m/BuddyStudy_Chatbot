import os

import streamlit as st
from groq import (
    APIConnectionError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

# ---------- Config ----------
MODEL="openai/gpt-oss-120b"
MAX_TOKENS = 1024
MAX_HISTORY_MESSAGES = 20  # only the last N messages are sent to the model

SYSTEM_PROMPT = """You are Buddy, a friendly and expert Study Buddy AI assistant.
Your job is to:
1. Explain complex topics simply, with real-life examples and analogies
2. Use code examples when explaining programming topics
3. Quiz the student when asked (one question at a time, then wait for the answer)
4. Be encouraging and motivating
5. Reply in the same language AND script the student uses (English, Hindi, or Hinglish).
   If the student writes Hinglish (Hindi in English letters), reply in Hinglish.
   Do not add English translations in brackets.

Keep responses clear, structured, and engaging. Never call yourself "Study Buddy Buddy" - your name is just Buddy."""

GREETING = "Hey! Main Buddy hoon, tumhara Study Buddy 📚 Aaj kya padhna hai?"

st.set_page_config(page_title="Study Buddy", page_icon="📚")


# ---------- API client ----------
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_resource
def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


api_key = get_api_key()
if not api_key:
    st.error(
        "GROQ_API_KEY not found. Add it in `.streamlit/secrets.toml` "
        "(locally) or in the Streamlit Cloud app's Secrets settings."
    )
    st.stop()

client = get_client(api_key)


# ---------- Chat logic ----------
def stream_reply(history):
    """Yield the reply from Groq chunk by chunk."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history[-MAX_HISTORY_MESSAGES:]],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def reset_chat():
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]


if "messages" not in st.session_state:
    reset_chat()

# ---------- UI ----------
st.title("📚 Study Buddy")
st.caption("Powered by Groq and LLaMA")

with st.sidebar:
    st.header("Options")
    st.button("🔄 New chat", on_click=reset_chat, use_container_width=True)
    st.markdown("---")
    st.markdown(
        "**Try asking:**\n"
        "- Explain recursion simply\n"
        "- Mujhe DBMS normalization samjhao\n"
        "- Quiz me on Python lists\n"
        "- Interview ke liye mock questions do"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Kuch bhi pucho..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except AuthenticationError:
            st.session_state.messages.pop()
            st.error("Invalid Groq API key. Please check your secret.")
        except RateLimitError:
            st.session_state.messages.pop()
            st.error("Rate limit reached. Thodi der baad try karo.")
        except APIConnectionError:
            st.session_state.messages.pop()
            st.error("Could not reach Groq. Check your internet connection.")
        except Exception as e:
            st.session_state.messages.pop()
            st.error(f"Something went wrong: {e}")

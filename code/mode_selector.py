import streamlit as st
import json

MODES = {
    "💻 Computer Mode": "computer",
    "🎵 Media Mode":    "media",
    "📽 Presentation":  "presentation"
}

st.set_page_config(page_title="Mode Selector", page_icon="🧠")
st.title("🧠 Gesture Control Mode Selector")

# Kullanıcı bir mod seçsin
choice = st.radio("Please choose a mode:", list(MODES.keys()))

if st.button("Save Mode"):
    mode_value = MODES[choice]
    with open("mod_config.json", "w") as f:
        json.dump({"mode": mode_value}, f)
    st.success(f"✅ '{choice}' saved! ({mode_value})")

st.caption("Select your gesture-control context here.")

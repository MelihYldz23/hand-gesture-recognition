import streamlit as st
import subprocess
import json
import os

# ─── Sidebar: Mode Selection ────────────────────────────────────────────────────
st.sidebar.title("Mode Selection")
modes = {
    "💻 Computer Mode": "computer",
    "🎵 Media Mode":    "media",
    "📽 Presentation":  "presentation"
}
sel = st.sidebar.radio("Select mode:", list(modes.keys()))
active_mode = modes[sel]

# Save the selected mode to a config file
with open("mod_config.json", "w") as f:
    json.dump({"mode": active_mode}, f)

st.markdown(f"## {sel}")

# ─── Media Mode ─────────────────────────────────────────────────────────────
if active_mode == "media":
    # Custom CSS
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50 !important;
            color: white !important;
            font-size: 1.2em;
            padding: 0.6em 2em;
            margin-bottom: 1em;
        }
        .info-text { font-size: 20px; margin-top: 1rem; }
        </style>
    """, unsafe_allow_html=True)

    vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

    # Video file uploader
    uploaded_file = st.file_uploader("📁 Select Video File", type=["mp4", "avi", "mkv", "mov", "wmv"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        temp_video_path = os.path.join("selected_video." + file_extension)

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        with open("video_path.json", "w") as f:
            json.dump({"path": os.path.abspath(temp_video_path)}, f)

        st.success(f"Video successfully uploaded: {uploaded_file.name}")

    if st.button("▶️ Start Camera"):
        subprocess.Popen(["python", "ai_mediamode_tahmin.py"], shell=True)
        st.success("Camera started! You can perform your hand gestures now.")

    st.markdown("### Media Mode Gestures - Command Mapping")
    st.markdown("""
        <div class="info-text">
            <p>- 0 ✋ → Play Video</p>
            <p>- 1 ✊ → Pause Video</p>
            <p>- 2 ☝ → Volume Up</p>
            <p>- 3 👇 → Volume Down</p>
            <p>- 5 🤙 → Stop Camera</p>
            <p>- 6 👉 → Seek Forward 10s</p>
            <p>- 7 👈 → Seek Backward 10s</p>
        </div>
    """, unsafe_allow_html=True)

# ─── Computer Mode ───────────────────────────────────────────────────────
elif active_mode == "computer":
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #4CAF50 !important;
            color: white !important;
            font-size: 1.2em;
            padding: 0.6em 2em;
        }
        .info-text { font-size: 20px; margin-top: 1rem; }
        </style>
    """, unsafe_allow_html=True)

    if st.button("▶️ Start Inference"):
        subprocess.Popen(["python", "ai_tahmin.py"], shell=True)
        st.success("Inference started! Check your OpenCV window.")

    st.markdown("### Computer Mode Gestures - Command Mapping")
    st.markdown("""
        <div class="info-text">
            <p>- 0 ✋ → Unmute</p>
            <p>- 1 ✊ → Mute</p>
            <p>- 2 ☝ → Brightness Up (100%)</p>
            <p>- 3 👇 → Brightness Down (0%)</p>
            <p>- 4 ✌ → Screenshot (saved in your Pictures/Screenshots folder)</p>
            <p>- 5 🤙 → Close the app</p>
        </div>
    """, unsafe_allow_html=True)

# ─── Presentation Mode ───────────────────────────────────────────────────────
else:
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FF5733 !important;
            color: white !important;
            font-size: 1.2em;
            padding: 0.6em 2em;
        }
        .info-text { font-size: 20px; margin-top: 1rem; }
        </style>
    """, unsafe_allow_html=True)

    uploaded_pptx = st.file_uploader("📁 Select Slide File (.pptx)", type=["pptx"])

    if uploaded_pptx is not None:
        pptx_path = os.path.join("selected_presentation.pptx")

        with open(pptx_path, "wb") as f:
            f.write(uploaded_pptx.read())

        with open("presentation_path.json", "w") as f:
            json.dump({"path": os.path.abspath(pptx_path)}, f)

        st.success(f"Presentation successfully uploaded: {uploaded_pptx.name}")
        st.info("Presentation file saved. Click below to start.")

    if st.button("🎬 Start Presentation (with Camera)"):
        try:
            pptx_full_path = os.path.abspath("selected_presentation.pptx")
            os.startfile(pptx_full_path)
            st.success("Presentation opened!")
        except Exception as e:
            st.error(f"Error opening presentation: {e}")

        subprocess.Popen(["python", "ai_presentation_tahmin.py"], shell=True)
        st.success("Camera started! You can now control your presentation with gestures.")

    st.markdown("### Presentation Mode Gestures - Command Mapping")
    st.markdown("""
        <div class="info-text">
            <p>- 0 ✋ → Start Presentation (F5)</p>
            <p>- 1 ✊ → Stop Presentation (ESC)</p>
            <p>- 6 👉 → Next Slide</p>
            <p>- 7 👈 → Previous Slide</p>
            <p>- 5 🤙 → Stop Camera and Close Presentation</p>
        </div>
    """, unsafe_allow_html=True)


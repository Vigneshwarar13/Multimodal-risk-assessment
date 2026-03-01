import streamlit as st
import os
import sys
import tempfile

# Add project root to Python path for imports to work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "flv", "wmv", "webm"]

# Set Streamlit config for larger uploads
st.set_page_config(
    page_title="Coercion Risk Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure Streamlit for larger file uploads
if not os.getenv("STREAMLIT_SERVER_MAXUPLOADSIZE"):
    os.environ["STREAMLIT_SERVER_MAXUPLOADSIZE"] = "500"


def main():
    st.title("🔍 Multimodal Coercion Risk Detection")
    st.write("Upload a verification video. The system analyzes speech and emotions to detect coercion.")

    with st.sidebar:
        st.info(
            "📋 **Upload Instructions**\n\n"
            "1. Select or drag-drop a video file\n"
            "2. Click 'Run Analysis'\n\n"
            "**File Size Limit:** up to 500 MB\n"
        )

    col1 = st.container()

    with col1:
        st.subheader("📹 Upload Verification Video")
        st.caption(f"Supported formats: {', '.join(VIDEO_FORMATS)}")
        st.caption("Max size: 500 MB")
        video_file = st.file_uploader("Upload video", type=VIDEO_FORMATS, key="video", accept_multiple_files=False)
        if video_file:
            file_size_mb = video_file.size / (1024 * 1024)
            st.caption(f"✅ Loaded: {video_file.name} ({file_size_mb:.1f} MB)")

    st.divider()
    run = st.button("🔍 Run Analysis", use_container_width=True, type="primary")

    if run:
        result = None

        if video_file is None:
            st.error("❌ Please upload a verification video.")
            return

        progress_container = st.container()

        try:
            from multimodal_coercion.engine.engine import verify_video
            with progress_container:
                st.info("⏳ Processing video: extracting audio, transcribing, and analyzing...")
                file_ext = os.path.splitext(video_file.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                result = verify_video(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                st.success("✅ Analysis complete")
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
            return

        if result:
            st.divider()
            st.subheader("📊 Analysis Results")

            # Unified scores
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🧠 Speech Intent Score", f"{int(round(100 * float(result.get('speech_intent_score', 0.0))))}")
            with c2:
                st.metric("🙂 Emotion Score", f"{int(round(100 * float(result.get('emotion_score', 0.0))))}")
            with c3:
                st.metric("🔊 Voice Stress", f"{int(round(100 * float(result.get('voice_stress_score', 0.0))))}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🧭 Willingness Score", f"{result.get('willingness_score', 0)}")
            with col2:
                label = str(result.get("final_decision", "AVERAGE")).upper()
                risk_emoji = "🔴" if label == "WORST" else "🟡" if label == "AVERAGE" else "🟢"
                st.metric(f"{risk_emoji} Final Status", label.title())

            st.subheader("📝 Transcript")
            st.write(result.get("transcript", ""))

            st.subheader("🙂 Emotion Summary")
            st.write(result.get("emotion_summary", ""))

            st.subheader("Recommendation")
            rec = result.get("recommended_action", "")
            if label == "GOOD":
                st.success(rec or "Successfully verified")
            elif label == "AVERAGE":
                st.warning(rec or "Reupload video")
            else:
                st.error(rec or "Restart verification and re-check documentation from beginning")


if __name__ == "__main__":
    main()

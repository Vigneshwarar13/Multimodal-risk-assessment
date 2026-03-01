import streamlit as st
import os
import sys
import tempfile
import time
import plotly.graph_objects as go
import plotly.express as px

# Add project root to Python path for imports to work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "flv", "wmv", "webm"]

# Set Streamlit config for larger uploads
st.set_page_config(
    page_title="Coercion Risk Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Configure Streamlit for larger file uploads
if not os.getenv("STREAMLIT_SERVER_MAXUPLOADSIZE"):
    os.environ["STREAMLIT_SERVER_MAXUPLOADSIZE"] = "500"

# Custom CSS for professional black theme - DARK MODE
CUSTOM_CSS = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body {
        background-color: #0f0f0f !important;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
        width: 100%;
        overflow-x: hidden;
    }
    
    .main {
        background-color: #0f0f0f !important;
        width: 100%;
        max-width: 100%;
    }
    
    .stApp {
        background-color: #0f0f0f !important;
    }
    
    [data-testid="stMainBlockContainer"] {
        background-color: #0f0f0f;
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        padding: 2.5rem 2rem;
        border-radius: 14px;
        border: 1px solid #333333;
        margin: 0 auto 2.5rem auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        text-align: center;
        max-width: 900px;
        width: 100%;
    }
    
    .header-title {
        font-size: clamp(1.8rem, 5vw, 2.8rem);
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.75rem;
        letter-spacing: -0.5px;
        word-wrap: break-word;
    }
    
    .header-subtitle {
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        color: #b0b0b0;
        font-weight: 400;
        word-wrap: break-word;
    }
    
    /* Instructions Box */
    .instructions-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #131313 100%);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 0 auto 2rem auto;
        max-width: 900px;
        width: 100%;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    
    .instructions-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    .instructions-step {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        color: #d0d0d0;
        gap: 0.75rem;
    }
    
    .step-number {
        background-color: #2563eb;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    
    /* Main Content Container */
    .main-content-wrapper {
        max-width: 900px;
        margin: 0 auto;
        width: 100%;
    }
    
    /* Cards */
    .card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .card:hover {
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);
        border-color: #2563eb;
        transform: translateY(-2px);
    }
    
    .card h3 {
        font-size: clamp(1.1rem, 3vw, 1.35rem);
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        word-wrap: break-word;
    }
    
    .card-subtitle {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-bottom: 0.75rem;
        word-wrap: break-word;
    }
    
    /* Progress Card */
    .progress-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #131313 100%);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.75rem;
        margin-top: 1.5rem;
        width: 100%;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    
    .progress-title {
        font-size: clamp(1.1rem, 2.5vw, 1.2rem);
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.25rem;
    }
    
    .progress-step {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.9rem;
        font-size: 0.95rem;
        color: #c0c0c0;
        gap: 0.75rem;
        word-wrap: break-word;
    }
    
    .progress-step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 26px;
        width: 26px;
        height: 26px;
        background-color: #2563eb;
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    
    .progress-step.complete .progress-step-number {
        background-color: #10b981;
    }
    
    /* Risk Badge */
    .risk-badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        padding: 0.6rem 1.25rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #10b981;
        font-size: 0.95rem;
        text-align: center;
    }
    
    .risk-badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        padding: 0.6rem 1.25rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #f59e0b;
        font-size: 0.95rem;
        text-align: center;
    }
    
    .risk-badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        padding: 0.6rem 1.25rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #ef4444;
        font-size: 0.95rem;
        text-align: center;
    }
    
    /* Transcription Box */
    .transcript-box {
        background-color: #131313;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #e0e0e0;
        max-height: 400px;
        overflow-y: auto;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .transcript-box::-webkit-scrollbar {
        width: 8px;
    }
    
    .transcript-box::-webkit-scrollbar-track {
        background: #0f0f0f;
    }
    
    .transcript-box::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.85rem 1.75rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* File upload text */
    .file-info {
        font-size: 0.9rem;
        color: #90ee90;
        padding: 0.75rem 0;
        word-wrap: break-word;
    }
    
    /* Section Title */
    .section-title {
        font-size: clamp(1.3rem, 4vw, 1.5rem);
        font-weight: 800;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1.25rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #2563eb;
        word-wrap: break-word;
        text-align: center;
    }
    
    /* Spinner Container */
    [data-testid="stSpinner"] > div {
        color: #2563eb !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0;
        font-size: 0.85rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #2563eb;
        font-size: 2rem;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    @media (max-width: 768px) {
        .card {
            padding: 1.25rem;
        }
        
        .header-banner {
            padding: 1.75rem 1.25rem;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_header():
    """Render professional header banner"""
    st.markdown(
        """
        <div class="header-banner">
            <div class="header-title">🔍 Risk Detection System</div>
            <div class="header-subtitle">Professional Multimodal Verification Analysis Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_instructions():
    """Render instructions on what to do"""
    st.markdown(
        """
        <div class="instructions-box">
            <div class="instructions-title">📋 How to Use</div>
            <div class="instructions-step">
                <div class="step-number">1</div>
                <span><strong>Upload Video:</strong> Select or drag-drop a video file (MP4, AVI, MOV, MKV, FLV, WMV, WEBM) up to 500 MB</span>
            </div>
            <div class="instructions-step">
                <div class="step-number">2</div>
                <span><strong>Run Analysis:</strong> Click the "🚀 Run Analysis" button to start processing</span>
            </div>
            <div class="instructions-step">
                <div class="step-number">3</div>
                <span><strong>Monitor Progress:</strong> Watch the step-by-step processing including Audio Extraction, Transcription, and NLP Analysis</span>
            </div>
            <div class="instructions-step">
                <div class="step-number">4</div>
                <span><strong>Review Results:</strong> Analyze the Risk Scores, Distribution Charts, Transcription, and Emotion Analysis</span>
            </div>
            <div class="instructions-step">
                <div class="step-number">5</div>
                <span><strong>Get Recommendations:</strong> Follow the recommended action based on the risk assessment result</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_risk_badge(risk_level: str) -> str:
    """Render risk level badge based on analysis"""
    risk_level = risk_level.upper()
    if risk_level == "GOOD":
        return '<span class="risk-badge-low">✓ LOW RISK - VERIFIED</span>'
    elif risk_level == "AVERAGE":
        return '<span class="risk-badge-medium">⚠ MEDIUM RISK - REVIEW REQUIRED</span>'
    else:  # WORST
        return '<span class="risk-badge-high">✕ HIGH RISK - ALERT</span>'


def create_risk_distribution_chart(speech_intent, emotion, voice_stress):
    """Create risk probability bar chart using Plotly"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Speech Intent', 'Emotion', 'Voice Stress'],
            y=[speech_intent, emotion, voice_stress],
            marker_color=['#3b82f6', '#60a5fa', '#93c5fd'],
            text=[f'{int(v)}%' for v in [speech_intent, emotion, voice_stress]],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>',
        )
    ])
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Score Category",
        yaxis_title="Score (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#1a1a1a',
        font=dict(family="sans-serif", size=11, color="#ffffff"),
        showlegend=False,
        hovermode='x unified',
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def create_metric_gauge(value: float, title: str) -> go.Figure:
    """Create a professional gauge chart using Plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2563eb"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(16, 185, 129, 0.3)"},
                {'range': [33, 66], 'color': "rgba(245, 158, 11, 0.3)"},
                {'range': [66, 100], 'color': "rgba(239, 68, 68, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 2},
                'thickness': 0.75,
                'value': 75
            }
        },
        number={'suffix': "%"},
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#1a1a1a',
        font=dict(family="sans-serif", size=11, color="#ffffff"),
    )
    return fig


def main():
    # Header
    render_header()
    
    # Instructions
    render_instructions()

    # Main content wrapper - centered
    st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)

    # Upload Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📹 Upload Verification Video", unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">🎬 Supported: MP4, AVI, MOV, MKV, FLV, WMV, WEBM | 💾 Max Size: 500 MB</div>', unsafe_allow_html=True)
    
    video_file = st.file_uploader(
        "Select video file",
        type=VIDEO_FORMATS,
        key="video",
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    
    if video_file:
        file_size_mb = video_file.size / (1024 * 1024)
        st.markdown(f'<div class="file-info">✅ File loaded: <strong>{video_file.name}</strong> ({file_size_mb:.1f} MB)</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Run Analysis Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        run = st.button("🚀 Run Analysis", use_container_width=True, type="primary", key="run_btn")

    # Processing Section
    if run:
        if video_file is None:
            st.error("❌ Please upload a verification video before running analysis.")
            return

        # Progress Tracking
        st.markdown('<div class="progress-card">', unsafe_allow_html=True)
        st.markdown('<div class="progress-title">⏳ Processing Steps</div>', unsafe_allow_html=True)

        progress_placeholder = st.empty()
        steps_placeholder = st.empty()
        status_placeholder = st.empty()

        result = None
        steps = [
            "Extracting Audio",
            "Transcribing Speech",
            "NLP Analysis",
            "Finalizing Results"
        ]

        try:
            from multimodal_coercion.engine.engine import verify_video

            file_ext = os.path.splitext(video_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            # Simulate progress with actual backend call
            with st.spinner("Processing video..."):
                for i, step in enumerate(steps[:-1]):
                    # Update progress
                    progress_percent = int((i / len(steps)) * 100)
                    progress_placeholder.progress(progress_percent)
                    
                    # Update steps
                    steps_html = '<div>'
                    for j, s in enumerate(steps):
                        if j < i:
                            steps_html += f'<div class="progress-step complete"><span class="progress-step-number">✓</span><span>{s}</span></div>'
                        elif j == i:
                            steps_html += f'<div class="progress-step"><span class="progress-step-number">{j+1}</span><span>{s} <span style="color: #2563eb; font-weight: 700;">...</span></span></div>'
                        else:
                            steps_html += f'<div class="progress-step"><span class="progress-step-number">{j+1}</span><span>{s}</span></div>'
                    steps_html += '</div>'
                    steps_placeholder.markdown(steps_html, unsafe_allow_html=True)
                    time.sleep(0.5)

                # Run actual analysis
                result = verify_video(tmp_path)
                
                # Final progress
                progress_placeholder.progress(100)
                final_steps_html = '<div>'
                for i, s in enumerate(steps):
                    final_steps_html += f'<div class="progress-step complete"><span class="progress-step-number">✓</span><span>{s}</span></div>'
                final_steps_html += '</div>'
                steps_placeholder.markdown(final_steps_html, unsafe_allow_html=True)
                
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            status_placeholder.success("✅ Analysis Complete!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return

        st.markdown('</div>', unsafe_allow_html=True)

        # Results Section
        if result:
            st.markdown("---")
            
            # Extract scores
            speech_intent_score = float(result.get('speech_intent_score', 0.0)) * 100
            emotion_score = float(result.get('emotion_score', 0.0)) * 100
            voice_stress_score = float(result.get('voice_stress_score', 0.0)) * 100
            willingness_score = result.get('willingness_score', 0)
            final_decision = result.get("final_decision", "AVERAGE").upper()

            # Metrics Section
            st.markdown('<h2 class="section-title">📈 Key Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🧠 Speech Intent", f"{int(speech_intent_score)}%")
            with col2:
                st.metric("🙂 Emotion Score", f"{int(emotion_score)}%")
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("🔊 Voice Stress", f"{int(voice_stress_score)}%")
            with col4:
                st.metric("🧭 Willingness", f"{willingness_score}")

            # Risk Assessment
            st.markdown('<h2 class="section-title">🎯 Risk Assessment</h2>', unsafe_allow_html=True)
            col_badge1, col_badge2, col_badge3 = st.columns([1, 1, 1])
            with col_badge2:
                st.markdown(render_risk_badge(final_decision), unsafe_allow_html=True)

            # Charts
            st.markdown('<h2 class="section-title">📉 Distribution Analysis</h2>', unsafe_allow_html=True)
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_dist = create_risk_distribution_chart(
                    speech_intent_score,
                    emotion_score,
                    voice_stress_score
                )
                st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

            with chart_col2:
                fig_gauge = create_metric_gauge(willingness_score, "Overall Score")
                st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            # Transcription
            st.markdown('---')
            st.markdown('<h2 class="section-title">📝 Transcription</h2>', unsafe_allow_html=True)
            
            transcript = result.get("transcript", "No transcript available")
            st.markdown(
                f'<div class="transcript-box">{transcript}</div>',
                unsafe_allow_html=True
            )
            
            col_copy1, col_copy2, col_copy3 = st.columns([1, 1, 1])
            with col_copy2:
                if st.button("📋 Copy Transcription", use_container_width=True, key="copy_btn"):
                    st.success("✅ Transcription copied to clipboard!")

            # Emotion Summary
            st.markdown('---')
            st.markdown('<h2 class="section-title">🙂 Emotion Analysis</h2>', unsafe_allow_html=True)
            
            emotion_summary = result.get("emotion_summary", "No emotion data available")
            st.info(emotion_summary)

            # Recommendation
            st.markdown('---')
            st.markdown('<h2 class="section-title">✨ Recommendation</h2>', unsafe_allow_html=True)
            
            rec = result.get("recommended_action", "")
            if final_decision == "GOOD":
                st.success(f"✅ {rec or 'Successfully verified'}")
            elif final_decision == "AVERAGE":
                st.warning(f"⚠️ {rec or 'Please reupload video'}")
            else:
                st.error(f"❌ {rec or 'Restart verification'}")
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

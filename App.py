import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="Hands-on-Wheel Monitoring System",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background-color:#0F172A;
}

h1,h2,h3,h4,p,label{
    color:white !important;
}

hr{
    border:1px solid #334155;
}

div[data-testid="stFileUploader"]{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Title ---------------- #

st.markdown("""
<h1 style='text-align:center;color:#38BDF8;'>
🚗 Hands-on-Wheel Monitoring System
</h1>

<h3 style='text-align:center;color:white;'>
for Autonomous Vehicles
</h3>

<hr>
""", unsafe_allow_html=True)

st.write(
    "Upload a driver image to detect whether the driver's hands are on the steering wheel."
)

# ---------------- Load Model ---------------- #

model = YOLO("best.pt")

# ---------------- Upload ---------------- #

uploaded_file = st.file_uploader(
    "📤 Upload Driver Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp.name)

    results = model.predict(
        source=temp.name,
        conf=0.5
    )

    plotted = results[0].plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🤖 Detection Result")
        st.image(plotted, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("📊 Detection Status")

    if len(results[0].boxes) > 0:

        confidence = float(results[0].boxes.conf[0])

        st.markdown(f"""
        <div style="background:#1E293B;
        padding:25px;
        border-radius:15px;
        border-left:8px solid #22C55E;">

        <h2 style="color:#22C55E;">
        🟢 Hands Detected
        </h2>

        <h3 style="color:white;">
        Confidence : {confidence:.1%}
        </h3>

        <p style="font-size:20px;color:white;">
        ✔ The driver's hands are detected on the steering wheel.
        </p>

        <p style="font-size:18px;color:#CBD5E1;">
        The driver appears ready to take control if the autonomous system requests it.
        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="background:#1E293B;
        padding:25px;
        border-radius:15px;
        border-left:8px solid #EF4444;">

        <h2 style="color:#EF4444;">
        🔴 No Hands Detected
        </h2>

        <h3 style="color:white;">
        Confidence : 0%
        </h3>

        <p style="font-size:20px;color:white;">
        ⚠ No hands were detected on the steering wheel.
        </p>

        <p style="font-size:18px;color:#CBD5E1;">
        Please place your hands on the steering wheel. The driver may not be ready to take control.
        </p>

        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<hr>

<center>

<p style="color:#94A3B8;font-size:16px;">
Graduation Project | AI Driver Monitoring System
</p>

</center>
""", unsafe_allow_html=True)
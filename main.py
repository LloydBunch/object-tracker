import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import numpy as np
from collections import defaultdict
import threading
import io
import zipfile
from datetime import datetime
from pathlib import Path

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #ffffff; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        border: 1px solid #e94560;
        box-shadow: 0 4px 20px rgba(233,69,96,0.3);
    }
    .main-title {
        font-size: 2.5rem; font-weight: 700; color: #ffffff;
        margin: 0; letter-spacing: 2px;
    }
    .main-subtitle { color: #a0aec0; font-size: 1rem; margin-top: 0.5rem; }

    .status-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: rgba(233,69,96,0.1); border: 1px solid #e94560;
        border-radius: 20px; padding: 0.3rem 1rem;
        font-size: 0.85rem; color: #e94560;
    }
    .status-dot {
        width: 8px; height: 8px; background: #e94560;
        border-radius: 50%; animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.3; }
        100% { opacity: 1; }
    }

    /* Video container */
    .video-container {
        border: 2px solid #e94560; border-radius: 12px;
        overflow: hidden; box-shadow: 0 8px 30px rgba(233,69,96,0.3);
    }

    /* Capture button */
    div[data-testid="stButton"].capture-btn > button {
        background: linear-gradient(135deg, #e94560, #c73652) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
        letter-spacing: 2px !important; font-size: 0.9rem !important;
        box-shadow: 0 4px 15px rgba(233,69,96,0.4) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"].capture-btn > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(233,69,96,0.6) !important;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0f3460, #16213e) !important;
        color: white !important; border: 1px solid #e94560 !important;
        border-radius: 8px !important; font-weight: 600 !important;
        letter-spacing: 1px !important; transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #e94560, #c73652) !important;
        transform: translateY(-2px) !important;
    }

    /* Clear button */
    div[data-testid="stButton"].clear-btn > button {
        background: transparent !important; color: #e94560 !important;
        border: 1px solid #e94560 !important; border-radius: 8px !important;
        font-size: 0.8rem !important; letter-spacing: 1px !important;
    }
    div[data-testid="stButton"].clear-btn > button:hover {
        background: rgba(233,69,96,0.1) !important;
    }

    /* Sidebar */
    .sidebar-title {
        color: #e94560; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px;
        margin: 1rem 0 0.5rem 0; padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(233,69,96,0.2);
    }

    /* Capture info card */
    .capture-info {
        background: rgba(233,69,96,0.05);
        border: 1px solid rgba(233,69,96,0.2);
        border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem;
    }
    .capture-info-row {
        display: flex; justify-content: space-between;
        font-size: 0.78rem; padding: 0.2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .capture-info-row:last-child { border-bottom: none; }
    .capture-info-label { color: #718096; }
    .capture-info-value { color: #e94560; font-weight: 600; }

    /* Gallery header */
    .gallery-header {
        color: #e94560; font-size: 0.85rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 0.8rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(233,69,96,0.3);
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    .stSlider label, .stSelectbox label {
        color: #a0aec0 !important; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Global shared store (survives reruns via cache_resource) ─────────────────
# cache_resource returns the SAME object every run → safe to mutate from any thread
@st.cache_resource
def get_shared_store():
    return {
        "latest_frame"   : None,   # numpy BGR
        "detection_count": 0,
        "tracked_ids"    : set(),
        "lock"           : threading.Lock(),
    }

@st.cache_resource
def get_track_history():
    """Persists between reruns; only touched by callback thread."""
    return defaultdict(list)


shared = get_shared_store()
track_history = get_track_history()

# ─── Disk save directory ──────────────────────────────────────────────────────
SAVE_DIR = Path("captured_frames")
SAVE_DIR.mkdir(exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ─── COCO classes ─────────────────────────────────────────────────────────────
COCO_CLASSES = {
    0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',
    5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',
    10:'fire hydrant',11:'stop sign',12:'parking meter',13:'bench',
    14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',
    20:'elephant',21:'bear',22:'zebra',23:'giraffe',24:'backpack',
    25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',
    30:'skis',31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',
    35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',
    39:'bottle',40:'wine glass',41:'cup',42:'fork',43:'knife',
    44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',
    49:'orange',50:'broccoli',51:'carrot',52:'hot dog',53:'pizza',
    54:'donut',55:'cake',56:'chair',57:'couch',58:'potted plant',
    59:'bed',60:'dining table',61:'toilet',62:'tv',63:'laptop',
    64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',
    69:'oven',70:'toaster',71:'sink',72:'refrigerator',73:'book',
    74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',
    79:'toothbrush',
}

def get_class_color(class_id: int):
    rng = np.random.default_rng(class_id)
    return tuple(rng.integers(50, 255, 3).tolist())


# ─── Session state defaults ───────────────────────────────────────────────────
for key, default in [
    ("saved_frames", []),
    ("capture_meta", None),
    ("last_capture_time", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div style="display:flex;align-items:center;gap:1rem;">
    <div>
      <div class="main-title">LIVE OBJECT DETECTION</div>
      <div class="main-subtitle">
          Real-time detection and tracking powered by YOLOv8
      </div>
    </div>
    <div style="margin-left:auto;">
      <div class="status-badge">
        <div class="status-dot"></div>SYSTEM ACTIVE
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;
                border-bottom:1px solid #2d3748;margin-bottom:1rem;">
        <div style="font-size:1.2rem;font-weight:700;
                    color:#e94560;letter-spacing:2px;">CONTROL PANEL</div>
        <div style="font-size:0.75rem;color:#a0aec0;margin-top:0.3rem;">
            Detection Configuration
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Detection Settings</div>',
                unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.5, 0.05,
        help="Minimum confidence score for object detection")

    st.markdown('<div class="sidebar-title">Tracking Settings</div>',
                unsafe_allow_html=True)
    show_trails     = st.toggle("Show Movement Trails", value=True)
    trail_length    = st.slider("Trail Length", 5, 60, 30, 5)
    show_labels     = st.toggle("Show Labels", value=True)
    show_confidence = st.toggle("Show Confidence", value=True)

    st.markdown('<div class="sidebar-title">Display Settings</div>',
                unsafe_allow_html=True)
    bbox_thickness = st.slider("Bounding Box Thickness", 1, 5, 2, 1)

    st.markdown('<div class="sidebar-title">Capture Settings</div>',
                unsafe_allow_html=True)
    capture_quality = st.select_slider(
        "JPEG Quality", options=[60, 70, 80, 90, 95, 100], value=95)
    add_timestamp_watermark = st.toggle("Add Timestamp Watermark", value=True)
    max_saved_frames = st.slider("Max Saved Frames", 5, 50, 20, 5)

    st.markdown("""
    <div style="margin-top:2rem;padding:1rem;
                background:rgba(233,69,96,0.05);
                border:1px solid rgba(233,69,96,0.2);border-radius:8px;">
        <div style="color:#e94560;font-weight:600;
                    font-size:0.85rem;margin-bottom:0.5rem;">MODEL INFO</div>
        <div style="color:#a0aec0;font-size:0.8rem;line-height:1.8;">
            Model: YOLOv8n<br>Classes: 80 COCO<br>
            Framework: Ultralytics<br>Tracker: BoT-SORT
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Video frame callback ─────────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]

    results = model.track(
        img, persist=True,
        conf=confidence_threshold,
        verbose=False,
        tracker="botsort.yaml",
    )

    annotated   = img.copy()
    det_count   = 0
    tracked_ids = set()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes     = results[0].boxes
        det_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf       = float(box.conf[0])
            cls_id     = int(box.cls[0])
            class_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")
            track_id   = int(box.id[0]) if box.id is not None else None

            if track_id is not None:
                tracked_ids.add(track_id)

            color = get_class_color(cls_id)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, bbox_thickness)

            # Corner accents
            cl = max(1, min(15, (x2 - x1) // 4, (y2 - y1) // 4))
            for px, py, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(annotated, (px, py), (px + dx * cl, py),
                         color, bbox_thickness + 1)
                cv2.line(annotated, (px, py), (px, py + dy * cl),
                         color, bbox_thickness + 1)

            # Movement trail
            if show_trails and track_id is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > trail_length:
                    track_history[track_id].pop(0)
                pts = track_history[track_id]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    tc    = tuple(int(c * alpha) for c in color)
                    th    = max(1, int(bbox_thickness * alpha))
                    cv2.line(annotated, pts[i - 1], pts[i], tc, th)

            # Label
            if show_labels:
                parts = []
                if track_id is not None:
                    parts.append(f"ID:{track_id}")
                parts.append(class_name)
                if show_confidence:
                    parts.append(f"{conf:.0%}")
                label = " | ".join(parts)

                (lw, lh), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly = max(y1 - 5, lh + 10)
                overlay = annotated.copy()
                cv2.rectangle(overlay,
                              (x1, ly - lh - 5),
                              (x1 + lw + 8, ly + baseline),
                              color, -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                cv2.putText(annotated, label, (x1 + 4, ly - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

    # Watermark
    cv2.putText(annotated,
                f"YOLOv8 | conf:{confidence_threshold:.0%} | det:{det_count}",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Write to shared store (thread-safe) ───────────────────────────────────
    with shared["lock"]:
        shared["latest_frame"]    = annotated.copy()
        shared["detection_count"] = det_count
        shared["tracked_ids"]     = tracked_ids.copy()

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── Layout columns ───────────────────────────────────────────────────────────
col_video, col_panel = st.columns([3, 1])

with col_video:
    # ── Stream ────────────────────────────────────────────────────────────────
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": {
                "width" : {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Capture bar ───────────────────────────────────────────────────────────
    cap_col1, cap_col2, cap_col3 = st.columns([2, 2, 1])

    with cap_col1:
        capture_clicked = st.button(
            "CAPTURE FRAME",
            key="capture_btn",
            help="Save the current annotated frame",
            use_container_width=True,
        )

    # Read shared stats (safe — only reading, lock still good practice)
    with shared["lock"]:
        live_det  = shared["detection_count"]
        live_ids  = len(shared["tracked_ids"])
        has_frame = shared["latest_frame"] is not None

    with cap_col2:
        st.markdown(f"""
        <div style="background:rgba(233,69,96,0.05);
                    border:1px solid rgba(233,69,96,0.2);
                    border-radius:8px;padding:0.6rem 1rem;
                    display:flex;gap:2rem;align-items:center;">
          <div style="text-align:center;">
            <div style="color:#e94560;font-size:1.4rem;font-weight:700;">
                {live_det}</div>
            <div style="color:#718096;font-size:0.7rem;
                        text-transform:uppercase;letter-spacing:1px;">
                Detections</div>
          </div>
          <div style="text-align:center;">
            <div style="color:#e94560;font-size:1.4rem;font-weight:700;">
                {live_ids}</div>
            <div style="color:#718096;font-size:0.7rem;
                        text-transform:uppercase;letter-spacing:1px;">
                Tracked IDs</div>
          </div>
          <div style="text-align:center;">
            <div style="color:#e94560;font-size:1.4rem;font-weight:700;">
                {len(st.session_state.saved_frames)}</div>
            <div style="color:#718096;font-size:0.7rem;
                        text-transform:uppercase;letter-spacing:1px;">
                Saved</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with cap_col3:
        if st.session_state.saved_frames:
            last = st.session_state.saved_frames[-1]
            st.download_button(
                label="LAST",
                data=last["bytes"],
                file_name=last["filename"],
                mime="image/jpeg",
                use_container_width=True,
                key="dl_last",
            )

    # ── Stream-active guard + capture logic ───────────────────────────────────
    stream_running = ctx.state.playing if ctx and ctx.state else False

    if capture_clicked:
        if not stream_running:
            st.warning("Start the camera stream first, then click Capture.")
        else:
            with shared["lock"]:
                frame_to_save = (shared["latest_frame"].copy()
                                 if shared["latest_frame"] is not None else None)
                snap_det  = shared["detection_count"]
                snap_ids  = shared["tracked_ids"].copy()

            if frame_to_save is None:
                st.warning("Frame not ready yet — wait a moment and try again.")
            else:
                save_img = frame_to_save.copy()
                now      = datetime.now()

                if add_timestamp_watermark:
                    wh, ww = save_img.shape[:2]
                    ts_str = now.strftime("%Y-%m-%d  %H:%M:%S")
                    (tw, th_), _ = cv2.getTextSize(
                        ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    overlay = save_img.copy()
                    cv2.rectangle(overlay,
                                  (0, wh - th_ - 16), (ww, wh),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.55, save_img, 0.45, 0, save_img)
                    cv2.putText(save_img, ts_str, (10, wh - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 200, 200), 1, cv2.LINE_AA)

                encode_params = [cv2.IMWRITE_JPEG_QUALITY, capture_quality]
                ok, buf = cv2.imencode(".jpg", save_img, encode_params)

                if ok:
                    filename  = f"capture_{now.strftime('%Y%m%d_%H%M%S_%f')[:21]}.jpg"
                    img_bytes = buf.tobytes()

                    disk_path = SAVE_DIR / filename
                    disk_path.write_bytes(img_bytes)

                    entry = {
                        "filename"  : filename,
                        "bytes"     : img_bytes,
                        "timestamp" : now.strftime("%Y-%m-%d %H:%M:%S"),
                        "detections": snap_det,
                        "tracked"   : len(snap_ids),
                        "path"      : str(disk_path),
                        "size_kb"   : round(len(img_bytes) / 1024, 1),
                    }
                    st.session_state.saved_frames.append(entry)
                    st.session_state.capture_meta = entry

                    # Prune oldest beyond max
                    while len(st.session_state.saved_frames) > max_saved_frames:
                        oldest = st.session_state.saved_frames.pop(0)
                        p = Path(oldest["path"])
                        if p.exists():
                            p.unlink()

                    st.success(f"Frame saved — {filename}")
                    st.rerun()
                else:
                    st.error("Encoding failed. Please try again.")


# ─── Right panel ──────────────────────────────────────────────────────────────
with col_panel:

    if st.session_state.capture_meta:
        meta = st.session_state.capture_meta
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:1px solid #e94560;border-radius:10px;
                    padding:0.8rem;margin-bottom:1rem;">
          <div class="gallery-header">Last Capture</div>
        """, unsafe_allow_html=True)

        nparr       = np.frombuffer(meta["bytes"], np.uint8)
        preview_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, use_container_width=True)

        st.markdown(f"""
        <div class="capture-info">
          <div class="capture-info-row">
            <span class="capture-info-label">Time</span>
            <span class="capture-info-value" style="font-size:0.72rem;">
                {meta['timestamp']}</span>
          </div>
          <div class="capture-info-row">
            <span class="capture-info-label">Objects</span>
            <span class="capture-info-value">{meta['detections']}</span>
          </div>
          <div class="capture-info-row">
            <span class="capture-info-label">Tracked IDs</span>
            <span class="capture-info-value">{meta['tracked']}</span>
          </div>
          <div class="capture-info-row">
            <span class="capture-info-label">Size</span>
            <span class="capture-info-value">{meta['size_kb']} KB</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.download_button(
            label="DOWNLOAD CAPTURE",
            data=meta["bytes"],
            file_name=meta["filename"],
            mime="image/jpeg",
            use_container_width=True,
            key=f"dl_preview_{meta['filename']}",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:1px solid #2d3748;border-radius:10px;
                padding:1rem;margin-bottom:1rem;">
      <div style="color:#e94560;font-size:0.85rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:1px;
                  margin-bottom:0.8rem;">How It Works</div>
      <div style="color:#a0aec0;font-size:0.8rem;line-height:1.8;">
        <div style="margin-bottom:0.5rem;">1. Camera captures live video</div>
        <div style="margin-bottom:0.5rem;">2. YOLOv8 detects objects</div>
        <div style="margin-bottom:0.5rem;">3. BoT-SORT assigns unique IDs</div>
        <div style="margin-bottom:0.5rem;">4. Trails show movement paths</div>
        <div style="margin-bottom:0.5rem;">5. Labels show class &amp; confidence</div>
        <div>6. Click <b style="color:#e94560;">CAPTURE FRAME</b> to save</div>
      </div>
    </div>
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:1px solid #2d3748;border-radius:10px;padding:1rem;">
      <div style="color:#e94560;font-size:0.85rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:1px;
                  margin-bottom:0.8rem;">Tips</div>
      <div style="color:#a0aec0;font-size:0.8rem;line-height:1.8;">
        <div style="margin-bottom:0.5rem;">• Good lighting improves detection</div>
        <div style="margin-bottom:0.5rem;">• Lower confidence catches more objects</div>
        <div style="margin-bottom:0.5rem;">• Trails help follow fast movement</div>
        <div style="margin-bottom:0.5rem;">• Allow camera access when prompted</div>
        <div>• Captures are saved to disk automatically</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Gallery ──────────────────────────────────────────────────────────────────
if st.session_state.saved_frames:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #2d3748;padding-top:1.5rem;">
      <div style="color:#e94560;font-size:1rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:2px;
                  margin-bottom:1rem;">Saved Frames Gallery</div>
    </div>""", unsafe_allow_html=True)

    g1, g2, _ = st.columns([1, 1, 4])

    with g1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in st.session_state.saved_frames:
                zf.writestr(f["filename"], f["bytes"])
        zip_buf.seek(0)
        st.download_button(
            label="DOWNLOAD ALL (ZIP)",
            data=zip_buf.getvalue(),
            file_name=f"captures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_all_zip",
        )

    with g2:
        if st.button("CLEAR ALL CAPTURES",
                     use_container_width=True, key="clear_all"):
            for f in st.session_state.saved_frames:
                p = Path(f["path"])
                if p.exists():
                    p.unlink()
            st.session_state.saved_frames.clear()
            st.session_state.capture_meta = None
            st.rerun()

    # Thumbnail grid
    COLS = 4
    frames_rev = list(reversed(st.session_state.saved_frames))
    for row_items in [frames_rev[i:i+COLS]
                      for i in range(0, len(frames_rev), COLS)]:
        grid = st.columns(COLS)
        for col, item in zip(grid, row_items):
            with col:
                nparr   = np.frombuffer(item["bytes"], np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
                st.markdown(f"""
                <div style="background:rgba(233,69,96,0.05);
                            border:1px solid rgba(233,69,96,0.15);
                            border-radius:6px;padding:0.4rem 0.6rem;
                            font-size:0.72rem;color:#a0aec0;
                            line-height:1.7;margin-bottom:0.3rem;">
                  <div><span style="color:#718096;">Time:</span>
                       <span style="color:#e94560;">{item['timestamp']}</span></div>
                  <div><span style="color:#718096;">Objects:</span>
                       <span style="color:#e94560;">{item['detections']}</span>
                       &nbsp;|&nbsp;
                       <span style="color:#718096;">IDs:</span>
                       <span style="color:#e94560;">{item['tracked']}</span></div>
                  <div><span style="color:#718096;">Size:</span>
                       <span style="color:#e94560;">{item['size_kb']} KB</span></div>
                </div>""", unsafe_allow_html=True)
                st.download_button(
                    label="DOWNLOAD",
                    data=item["bytes"],
                    file_name=item["filename"],
                    mime="image/jpeg",
                    use_container_width=True,
                    key=f"dl_{item['filename']}",
                )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding:1rem;text-align:center;
            border-top:1px solid #2d3748;color:#4a5568;
            font-size:0.75rem;letter-spacing:1px;">
    POWERED BY YOLOV8 + STREAMLIT-WEBRTC
    | REAL-TIME OBJECT DETECTION &amp; TRACKING
    | FRAME CAPTURE &amp; GALLERY
</div>""", unsafe_allow_html=True)
import os
import sys
import subprocess

# 1. Use /tmp/ for marker tracking to avoid virtual environment PermissionError
headless_marker = "/tmp/.opencv_headless_patched"

if not os.path.exists(headless_marker):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless==4.10.0.84"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Write marker file to /tmp/
    with open(headless_marker, "w") as f:
        f.write("patched")

# 2. Point DeepFace home directory to local project root
os.environ["DEEPFACE_HOME"] = os.getcwd()

# 3. Safe imports
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace

# Page configuration
st.set_page_config(
    page_title="Group Face Verification", 
    page_icon="📸", 
    layout="wide"
)

# 2. Cache DeepFace model loading in RAM across app sessions
@st.cache_resource
def load_models():
    """Builds and keeps Facenet512 in memory so it doesn't reload on every button click."""
    _ = DeepFace.build_model("Facenet512")
    return True

# Initialize model cache on startup
with st.spinner("Initializing AI models into memory..."):
    load_models()

# Custom CSS for centered title and natural webcam styling
st.markdown(
    """
    <style>
    h1 {
        text-align: center;
        text-decoration: underline;            /* Enables the underline */
        text-decoration-color: #ff4757;       /* Changes line color */
        text-decoration-thickness: 3px;       /* Adjusts line thickness */
        text-underline-offset: 5px;            /* Adds space below the title */
        padding-bottom: 20px;                 /* Adds space below the title */
    }
    video {
        -webkit-transform: scaleX(1); 
        transform: scaleX(1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📸 Group Face Verification 📸")


def cosine_distance(source_representation, test_representation):
    """Calculates exact cosine distance between two 512-D vectors."""
    a = np.array(source_representation)
    b = np.array(test_representation)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


col1, col2 = st.columns([1, 1])

# --- 1. GROUP PHOTO SECTION ---
with col1:
    st.header("1. Master Group Photo")
    group_file = st.file_uploader("Upload Group Photo", type=["jpg", "jpeg", "png"], key="group_uploader")
    group_np = None
    if group_file:
        group_img = Image.open(group_file).convert('RGB')
        group_np = np.array(group_img)
        st.image(group_np, caption="Master Group Photo", width="stretch")

user_file = None

# --- 2. USER PHOTO SECTION ---
with col2:
    st.header("2. Provide Your Photo")
    tab1, tab2 = st.tabs(["📷 Option 1: Live Photo", "📁 Option 2: Static Image"])
    
    with tab1:
        camera_file = st.camera_input("Take a selfie")
        if camera_file:
            user_file = camera_file
            
    with tab2:
        static_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"], key="user_uploader")
        if static_file:
            user_img = Image.open(static_file).convert('RGB')
            st.image(np.array(user_img), caption="Your Photo", width=300)
            user_file = static_file

# --- 3. MATCHING & VISUALIZATION LOGIC ---
if group_file and user_file:
    st.divider()
    st.header("3. Verification Result")
    
    with open("temp_group.jpg", "wb") as f:
        f.write(group_file.getbuffer())
    with open("temp_user.jpg", "wb") as f:
        f.write(user_file.getbuffer())
        
    with st.spinner("Extracting facial features using RetinaFace..."):
        try:
            # Step A: Get 512-D vector representation of user selfie
            user_objs = DeepFace.represent(
                img_path="temp_user.jpg",
                model_name="Facenet512",
                detector_backend="retinaface",
                enforce_detection=False
            )
            
            if not user_objs:
                st.error("Could not detect a face in the provided user photo. Please try a clearer picture.")
                st.stop()
                
            user_embedding = user_objs[0]["embedding"]
            user_area = user_objs[0]["facial_area"]

            # Step B: Get 512-D vector representations of all faces in group photo
            group_objs = DeepFace.represent(
                img_path="temp_group.jpg",
                model_name="Facenet512",
                detector_backend="retinaface", 
                enforce_detection=False
            )
            
            st.info(f"Successfully processed **{len(group_objs)}** faces in the group photo.")
            
            group_img_cv = cv2.imread("temp_group.jpg")
            img_h, img_w, _ = group_img_cv.shape
            
            best_distance = 1.0
            best_box = None
            best_crop = None
            
            # Accepts scores <= 0.70 and rejects anything above 0.70
            MATCH_THRESHOLD = 0.70
            
            for face in group_objs:
                facial_area = face["facial_area"]
                group_embedding = face["embedding"]
                
                # Compute distance vector
                dist = cosine_distance(user_embedding, group_embedding)
                
                if dist < best_distance:
                    best_distance = dist
                    x = max(0, int(facial_area['x']))
                    y = max(0, int(facial_area['y']))
                    w = int(facial_area['w'])
                    h = int(facial_area['h'])
                    best_box = (x, y, w, h)
                    
                    crop = group_img_cv[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)]
                    if crop.size > 0:
                        best_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # --- RENDER RESULTS ---
            if best_box is not None and best_crop is not None:
                
                # ONLY draw green box and accept if score <= 0.70
                if best_distance <= MATCH_THRESHOLD:
                    st.success(f"🎉 Match Confirmed! (Distance Score: {best_distance:.3f})")
                    
                    x, y, w, h = best_box
                    annotated_group = group_img_cv.copy()
                    
                    # Draw thick green bounding box around matched face
                    cv2.rectangle(annotated_group, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    annotated_rgb = cv2.cvtColor(annotated_group, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_rgb, caption=f"Target Location (Distance Score: {best_distance:.3f})", width="stretch")
                    
                    st.subheader("🔍 Matched Face Comparison")
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        ux = max(0, int(user_area['x']))
                        uy = max(0, int(user_area['y']))
                        uw = int(user_area['w'])
                        uh = int(user_area['h'])
                        
                        user_cv = cv2.imread("temp_user.jpg")
                        u_h, u_w, _ = user_cv.shape
                        user_crop = user_cv[max(0, uy):min(u_h, uy + uh), max(0, ux):min(u_w, ux + uw)]
                        
                        if user_crop.size > 0:
                            st.image(cv2.cvtColor(user_crop, cv2.COLOR_BGR2RGB), caption="Your Extracted Face", width=220)
                        else:
                            st.image(Image.open("temp_user.jpg"), caption="Your Uploaded Photo", width=220)
                    
                    with res_col2:
                        st.image(best_crop, caption=f"Group Cutout (Score: {best_distance:.3f})", width=220)
                
                else:
                    # REJECT: Show standard group photo without green box
                    st.warning(f"❌ Match Rejected! Closest candidate score was {best_distance:.3f} (Must be <= {MATCH_THRESHOLD})")
                    st.image(group_np, caption="Master Group Photo (No Confident Match)", width="stretch")
                    
        except Exception as e:
            st.error(f"Error processing images: {e}")
        finally:
            for temp in ["temp_group.jpg", "temp_user.jpg"]:
                if os.path.exists(temp):
                    os.remove(temp)


import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from face_recognition import ( 
    face_detector,
    embedder,
    index,
    face_db,
    mask_classifier,
    cap_classifier,
    register_person,
    recognize_faces,
    train_accessory_detectors
)


st.set_page_config(page_title="Face Recognition System", layout="wide")


st.markdown("""
<style>
    .stApp {
        max-width: 1600px;
    }
    .stButton>button {
        width: 100%;
    }
    .stFileUploader>div>div>div>div {
        color: #000000;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


st.title("Face Recognition System")



confidence_threshold = st.sidebar.slider(
    "Recognition Confidence Threshold",
    min_value=0.85,
    max_value=1.0,
    value=0.98,
    step=0.01
)


tab1, tab2, tab3 = st.tabs([
    "Register Faces", 
    "Recognize Faces", 
    "Database Info",
    
])

# Tab 1: Register Faces
with tab1:
    st.header("Register New Faces")
    col1, col2 = st.columns(2)
    
    with col1:
        reg_name = st.text_input("Person's Name", key="reg_name")
        reg_image = st.file_uploader(
            "Upload Face Image", 
            type=["jpg", "png", "jpeg"],
            key="reg_upload"
        )
        
        if st.button("Register Face"):
            if reg_name and reg_image:
                try:
                    
                    temp_path = f"temp_register_{reg_name}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(reg_image.getbuffer())
                    
                    
                    result = register_person(temp_path, reg_name)
                    st.success(result)
                    
                    
                    image = Image.open(temp_path)
                    st.image(image, caption=f"Registered: {reg_name}")
                    
                    
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")
            else:
                st.warning("Please provide both name and image")

# Tab 2: Recognize Faces
with tab2:
    st.header("Recognize Faces")
    rec_image = st.file_uploader(
        "Upload Image ", 
        type=["jpg", "png", "jpeg"],
        key="rec_upload"
    )
    
    if rec_image and st.button("Recognize Faces"):
        try:
            
            temp_path = "temp_recognize.jpg"
            with open(temp_path, "wb") as f:
                f.write(rec_image.getbuffer())
            
            
            results = recognize_faces(temp_path, confidence_threshold)
            
           
            image = cv2.imread(temp_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            for r in results:
                x1, y1, x2, y2 = r["box"]
                color = (0, 255, 0) if r["label"] != "Unknown" else (0, 0, 255)
                thickness = 2
                
                
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)
                
                
                label = f"{r['label']} ({r['confidence']:.1%})"
                if r.get("mask", False):
                    label += " [MASK]"
                if r.get("cap", False):
                    label += " [CAP]"
                
                
                cv2.putText(
                    image_rgb, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, 
                    color, 
                    thickness
                )
            
            
            st.image(image_rgb, caption="Recognition Results", use_column_width=True)
            
            
            st.subheader("Recognition Details")
            for i, r in enumerate(results, 1):
                st.write(f"**Face {i}**:")
                st.write(f"- Identity: {r['label']} (Confidence: {r['confidence']:.1%})")
                if r.get("mask", False):
                    st.write("- Detected: Wearing mask")
                if r.get("cap", False):
                    st.write("- Detected: Wearing cap")
            
            
            os.remove(temp_path)
        except Exception as e:
            st.error(f"Recognition failed: {str(e)}")

# Tab 3: Database Info
with tab3:
    st.header("Database Information")
    
    if not face_db["labels"]:
        st.warning("No faces registered in the database")
    else:
        st.write(f"Total registered faces: {len(face_db['labels'])}")
        st.write(f"Unique individuals: {len(set(face_db['labels']))}")
        
        st.subheader("Registered Persons")
        unique_persons = set(face_db["labels"])
        for person in unique_persons:
            count = face_db["labels"].count(person)
            st.write(f"- {person}: {count} face(s)")



# Run the app
if __name__ == "__main__":
    st.write("App is running!")
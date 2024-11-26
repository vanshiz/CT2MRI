import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import streamlit as st

st.set_page_config(
    page_title="DiagnoAid",  # Title of the tab
    page_icon="🔬",  # You can set an emoji or path to an image as an icon
    layout="wide"  # Optional: You can set the layout to "wide" if you want more space
)
# Load the pre-trained combined model
model = load_model('CombinedBrainTumorModel.h5')

# Class mapping function
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Preprocessing function
def preprocess_image(image_file):
    image = Image.open(image_file)
    image = image.convert("RGB")
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function for CT and MRI inputs
def getResult(ct_image_file, mri_image_file):
    try:
        ct_image = preprocess_image(ct_image_file)
        mri_image = preprocess_image(mri_image_file)
        
        # Perform prediction with both inputs
        result = model.predict([ct_image, mri_image])
        
        # Print the raw prediction result for debugging
        print(f"Raw prediction result: {result}")
        
        # Apply threshold (0.5) to classify as Tumor or No Tumor
        predicted_class = result[0][0]
        if predicted_class >= 0.5:
            return 1  # Tumor
        else:
            return 0  # No Tumor
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI

st.markdown(

    """
    <style>
    .sidebar .sidebar-content {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Main content adjustments
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 0rem;
        margin-top: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 35px;
        text-align: center;
        font-weight: bold;
        color: #FFF;
    }
    </style>
    <div class="sidebar-title">DiagnoAid</div>
    """,
    unsafe_allow_html=True
)



st.sidebar.header("Select an Option: ")

# Main Page
st.title("CT to MRI Conversion and Analysis")


# Sidebar Navigation Improvements
option = st.sidebar.selectbox(
    label="Navigation Options",  # Add an accessible label
    options=("Home", "Upload CT & MRI Scans", "Analysis"),
    label_visibility="collapsed"  # Hides the label from the UI
)
# File storage for uploaded scans
if "ct_file" not in st.session_state:
    st.session_state["ct_file"] = None
if "mri_file" not in st.session_state:
    st.session_state["mri_file"] = None

if option == "Home":
    st.markdown(
        """
        <br>
        <div style="text-align: center;">
            <h3>Welcome to DiagnoAid!</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1; padding-right: 20px;">
            <br><br>
                <p style="font-size: 1.2em; line-height: 1.6;">
                    DiagnoAid is your one-stop solution for advanced medical imaging analysis.
                    Our AI-powered platform assists doctors and patients in accurately diagnosing
                    conditions using cutting-edge machine learning models. From CT scans to MRI,
                    we've got you covered with fast, reliable, and secure results.
                </p>
            </div>
            
        </div>
        """,
        unsafe_allow_html=True
    )
    # st.image("sidebar.jpg", caption="DiagnoAid Illustration", width=300)

elif option == "Upload CT & MRI Scans":
    st.write("Upload your CT and MRI scans here.")

    # File uploader for CT scan
    ct_file = st.file_uploader("Upload CT Scan", type=["png", "jpg", "jpeg"])
    if ct_file:
        st.session_state["ct_file"] = ct_file
        st.success("CT scan uploaded successfully!")

    # File uploader for MRI scan
    mri_file = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"])
    if mri_file:
        st.session_state["mri_file"] = mri_file
        st.success("MRI scan uploaded successfully!")

    if ct_file and mri_file:
        st.success("Both files uploaded successfully! Proceed to the 'Analysis' tab for results.")

elif option == "Analysis":
    if st.session_state["ct_file"] and st.session_state["mri_file"]:
        if st.button("Analyze"):
            with st.spinner("Analyzing the scans..."):
                result = getResult(st.session_state["ct_file"], st.session_state["mri_file"])
                if isinstance(result, str) and result.startswith("Error"):
                    st.error(result)
                else:
                    diagnosis = get_className(result)
                    st.success(f"Diagnosis: {diagnosis}")
    else:
        st.warning("Please upload both CT and MRI scans in the 'Upload CT & MRI Scans' section.")

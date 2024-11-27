import os
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import cv2
from InstanceNormalization import InstanceNormalization
import tempfile


def convert_to_grayscale(image_file):
    # Read the image
    image = Image.open(image_file)
    image = np.array(image)
    
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Reshape to match model input shape (256, 256, 1)
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)
    return grayscale_image


st.set_page_config(
    page_title="DiagnoAid",  # Title of the tab
    page_icon="🔬",
    layout="wide"  
)

# Load the pre-trained models
combined_model = load_model('CombinedBrainTumorModel.h5')
cyclegan_model = load_model('cycleGAN.h5', custom_objects={'InstanceNormalization': InstanceNormalization})

# Class mapping function
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Preprocessing function for images
def preprocess_image(image_file, target_size=(64, 64)):
    image = Image.open(image_file)
    image = image.convert("RGB")
    image = image.resize(target_size)  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# CycleGAN generation function
def generate_mri_from_ct(ct_image_file):
    try:
        # Read and preprocess the image
        image = Image.open(ct_image_file).convert("RGB")
        image = image.resize((256, 256))  # Resize to CycleGAN's input size
        image = np.array(image) / 255.0  # Normalize pixel values

        # Convert to grayscale
        grayscale_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grayscale_image = np.expand_dims(grayscale_image, axis=-1)  # Add channel dimension

        # Prepare input data
        input_data = np.expand_dims(grayscale_image, axis=0)  # Add batch dimension

        # Generate MRI
        generated_mri = cyclegan_model.predict(input_data)
        generated_mri = np.squeeze(generated_mri, axis=0)  # Remove batch dimension

        # Ensure the generated MRI is within the correct range
        generated_mri = np.clip(generated_mri, 0, 1)  # Ensure values are within 0-1

        # Check the shape and ensure it is 2D or 3D
        if generated_mri.ndim == 2:
            generated_mri = np.expand_dims(generated_mri, axis=-1)  # Ensure it has a channel dimension if needed
        elif generated_mri.ndim == 3 and generated_mri.shape[-1] == 1:
            generated_mri = np.squeeze(generated_mri, axis=-1)  # Remove the channel if it is a grayscale image

        # Return the image as a uint8 array (0-255 range)
        return (generated_mri * 255).astype(np.uint8)
    except Exception as e:
        raise ValueError(f"Error generating MRI: {e}")

# Saving the generated MRI
def save_generated_mri(generated_mri):
    try:
        # Ensure the generated MRI has the correct shape and type for saving
        if generated_mri.ndim == 3 and generated_mri.shape[-1] == 1:
            generated_mri = np.squeeze(generated_mri, axis=-1)  # Remove the channel if grayscale
        if generated_mri.ndim == 2:
            # If grayscale image (2D), convert it to a proper format for saving
            generated_mri = np.expand_dims(generated_mri, axis=-1)

        # Save the generated MRI image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file_path = temp_file.name
        Image.fromarray(generated_mri).save(temp_file_path)
        return temp_file_path
    except Exception as e:
        raise ValueError(f"Error saving generated MRI: {e}")



# Prediction function for CT and MRI inputs
def getResult(ct_image_file, mri_image_file):
    try:
        ct_image = preprocess_image(ct_image_file)
        mri_image = preprocess_image(mri_image_file)
        
        # Perform prediction with both inputs
        result = combined_model.predict([ct_image, mri_image])
        
        # Apply threshold (0.5) to classify as Tumor or No Tumor
        predicted_class = result[0][0]
        if predicted_class >= 0.5:
            return 1  # Tumor
        else:
            return 0  # No Tumor
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI setup
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
    options=("Home", "Upload CT Scan", "Upload CT & MRI Scans", "Analysis"),
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

elif option == "Upload CT Scan":
    st.write("Upload your CT scan here to generate an MRI.")

    # File uploader for CT scan
    ct_file = st.file_uploader("Upload CT Scan", type=["png", "jpg", "jpeg"])
    if ct_file:
        st.session_state["ct_file"] = ct_file
        st.success("CT scan uploaded successfully!")

        # Generate MRI from uploaded CT scan
        st.write("Generating MRI from CT scan...")
        try:
            generated_mri = generate_mri_from_ct(ct_file)
            if isinstance(generated_mri, np.ndarray):
                # Store the generated MRI as a numpy array in session state
                st.session_state["generated_mri"] = generated_mri
                st.success("MRI generated successfully! Proceed to the 'Analysis' tab for prediction.")
            else:
                st.error("Error: MRI generation failed.")
        except Exception as e:
            st.error(f"Error during MRI generation: {e}")


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
    # Check if CT and either MRI (uploaded or generated) are available
    if (
        "ct_file" in st.session_state and st.session_state["ct_file"] is not None
        and (
            "mri_file" in st.session_state and st.session_state["mri_file"] is not None
            or "generated_mri" in st.session_state
        )
    ):
        if st.button("Analyze"):
            with st.spinner("Analyzing the scans..."):
                # Use the generated MRI if available; otherwise, use the uploaded MRI
                if "generated_mri" in st.session_state:
                    # Save generated MRI to a temporary file
                    generated_mri = st.session_state["generated_mri"]
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    temp_file_path = temp_file.name
                    try:
                        # Save generated MRI as an image
                        Image.fromarray(generated_mri).save(temp_file_path)
                        st.session_state["mri_file"] = temp_file_path  # Update the MRI file path
                    except Exception as e:
                        st.error(f"Error saving generated MRI: {e}")

                result = getResult(st.session_state["ct_file"], st.session_state["mri_file"])
                class_name = get_className(result)
                st.success(f"Prediction: {class_name}")
    else:
        st.warning("Please upload both the CT and MRI scans (or generate the MRI from CT scan).")

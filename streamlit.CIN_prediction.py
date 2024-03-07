## Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import tempfile
import os
from PIL import Image
import plotly.express as px

################
## Set up variables and load content

# Load an image from the file system (assumed to be in the same folder as this script).
logo = Image.open('logo.png')
gif_path = './waiting.gif'

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# Custom CSS to inject into the Streamlit interface 000000 v0076b6
button_css = """
<style>
div.stButton > button:first-child {
    color: white;
    background-color: #0076b6;
    border: none;
    border-radius: 5px;
    padding: 10px 24px;
    font-size: 16px;
    font-weight: bold;
    text-transform: uppercase;
    transition: background-color 0.3s, box-shadow 0.3s;
    cursor: pointer;
    box-shadow: 0 2px 4px 0 rgba(0,0,0,.2);
    display: block;
    margin: auto;
}
div.stButton > button:first-child:hover {
    background-color: #000000;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,.3);
}
div.stDownloadButton > button {
    /* Custom styles */
    color: #fff; /* Text color */
    background-color: #0076b6; /* Background color */
    border: none; /* Remove border */
    padding: 10px 24px; /* Padding inside the button */
    border-radius: 25px; /* Rounded corners */
    cursor: pointer; /* Cursor to pointer */
    font-size: 24x; /* Increase font size */
    transition: transform 0.2s, background-color 0.2s; /* Smooth transition for hover effects */
}

div.stDownloadButton > button:hover {
    background-color: #000000; /* Darker shade of green on hover */
    transform: scale(1.1); /* Scale button on hover */
    color: #FFFFFF; /* Text color changes on hover */
}
</style>
"""

# Inject custom CSS with Markdown
st.markdown(button_css, unsafe_allow_html=True)

# Display the logo at the top of the page.
st.image(logo, use_column_width=True)

# Section to explain how to use the app.
st.title("How to use this app")
st.write("This app is a showcase for MicronuclAI allows you to test our pretrained model with your own small test data.")

# Define input files
col1, col2, col3 = st.columns(3)
with col1:
    nuclei_image = st.file_uploader(
        "Upload a nuclear staining file:", accept_multiple_files=False, key="nuclei_image" )
    
with col2:
    mask_image = st.file_uploader(
        "Upload a nuclear mask file:", accept_multiple_files=False, key="nuclei_mask")
with col3:
    spacer = 3
    for _ in range(spacer):
        st.write("")  # These empty writes act as a spacer
    submit_button = st.button("Run the script", key="submit_button_key")

## Inference model file
model_file = "/Users/florian_wuennemann/1_Projects/Miguel_CIN_streamlit/cin_models/model_4.pt"

# Test 2: How to derive file paths from file uploads?
if nuclei_image is not None:
    temp_nucimage = tempfile.NamedTemporaryFile(
        prefix="nuclei.", dir=".")
    # temp_nucimage.name = 'nuclei_image.tif'
    temp_nucimage.write(nuclei_image.getbuffer())
if mask_image is not None:
    temp_maskimage = tempfile.NamedTemporaryFile(
        prefix="cin_inference.", dir=".")
    # temp_maskimage.name = 'mask.tif'
    temp_maskimage.write(mask_image.getbuffer())

################
## Run inference when user clicks run button
if 'count' not in st.session_state:
    st.session_state.count = 0

placeholder = st.empty()
# Run the inference script
if submit_button:
    with placeholder.container():
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.image(gif_path)
            subprocess.run([f"{sys.executable}",
                        "prediction2.py",
                        "-i", temp_nucimage.name,
                        "-m", temp_maskimage.name,
                        "-mod", model_file,
                        "-d", "mps",
                        "-o", "./results"])
            # Clear the GIF
            placeholder.empty()

        
    ## Generate output files
    output_prefix = temp_maskimage.name.split('/')[-1]
    pred_out = "cin_inference_predictions.csv"
    sum_out = "cin_inference_summary.csv"

    ## Result visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Micronuclei distribution")
        predictions = pd.read_csv("./results/"+pred_out)
        # Summarize the column micronuclei in predictions
        summary = predictions.groupby('micronuclei').size().reset_index(name='count')
        # Make the micronuclei column a factor variable
        summary['micronuclei'] = summary['micronuclei'].astype(str)
        
        # Create the Plotly bar chart
        fig = px.bar(summary, x='micronuclei', y='count', template= "simple_white")
        # Make the bar color 0,119,182
        fig.update_traces(marker_color='rgb(0,119,182)')

        # Display the figure in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)
        #st.write("This plot shows the distribution of micronuclei.")

        with open("./results/"+pred_out) as f:
            st.download_button('Download predictions', f, 'text/csv')

    with col2:
        st.subheader("Summary Table")
        micro_sum = pd.read_csv("./results/"+sum_out)
        st.table(micro_sum)
        with open("./results/"+sum_out) as s:
            st.download_button('Download summary', s, 'text/csv')

    # Scatterplot placeholder.
    st.header("Scatterplot of nuclei positions")
    # Generate test data with some points in x and y space with 20 points in each group and color them differently
    np.random.seed(0)
    x = np.random.randn(100)
    y = np.random.randn(100)
    group = np.random.choice(['A', 'B'], 100)
    nuclei_pos = pd.DataFrame(dict(x=x, y=y, group=group))
    nuclei_pos["group"] = nuclei_pos["group"].astype(str)
    # Plot a plotly plot for df nuclei_pos with x and y as x and y and group as color, with color palette tab10
    fig = px.scatter(nuclei_pos, x='x', y='y', color='group', template="simple_white")
    st.plotly_chart(fig, use_container_width=True)

    st.session_state.count += 1

#### Old code
# Test 1: Does the inference work when given hard paths for input files?
# WORKS!
# if st.button("Run the script"):
#     subprocess.run([f"{sys.executable}",
#                     "prediction2.py",
#                     "-i", "../data/24h_combined_071023-01-Stitching-03_s07.ome.tif",
#                     "-m", "../data/24h_combined_071023-01-Stitching-03_s07.MASK.16bit.tif",
#                     "-mod", model_file,
#                     "-d", "mps",
#                     "-o", "../results"])
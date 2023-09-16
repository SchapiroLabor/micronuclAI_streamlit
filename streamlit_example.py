import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
import tempfile

st.title('Perform micronulcei inference')
st.subheader(
    "This app will take as input an image with nuclei and a mask file for nuclei.")
st.subheader("It will produce two output files, one file containing predictions for each nucleus (*_predictions.csv) and one file containing a summary of the results (*_summary.csv)")

# Define input files
nuclei_image = st.file_uploader(
    "Upload a nuclear staining file:", accept_multiple_files=False, key="nuclei_image")
mask_image = st.file_uploader(
    "Upload a nuclear mask file:", accept_multiple_files=False, key="nuclei_mask")
model_file = "./model_4.pt"

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

if st.button("Run the script"):
    pipeline_ex = 1
    subprocess.run([f"{sys.executable}",
                    "prediction2.py",
                    "-i", temp_nucimage.name,
                    "-m", temp_maskimage.name,
                    "-mod", model_file,
                    "-d", "mps",
                    "-o", "../results"])
    # Get only the filename without path ffrom temp_maskimage and add _predictions.csv
    output_prefix = temp_maskimage.name.split('/')[-1]
    st.write(temp_maskimage.name)
    st.write(output_prefix)
    pred_out = "cin_inference_predictions.csv"
    sum_out = "cin_inference_summary.csv"

    # If the inference has been run once, show the download buttons
    if pipeline_ex == 1:
        with open("/results/"+pred_out) as f:
            st.download_button('Download predictions:', f, 'text/csv')
        with open("/results/"+sum_out) as s:
            st.download_button('Download summary:', s, 'text/csv')

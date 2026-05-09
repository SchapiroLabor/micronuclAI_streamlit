## Import libraries
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

################
## Set up variables and load content
ROOT = Path(__file__).absolute().parent
RESULTS = ROOT / "results"
GIF = ROOT / "animation_micronuclei.gif"
LOGO = ROOT / "logo.png"
TEST_DATA = ROOT / "testdata.tar.gz"
MODEL = ROOT / "micronuclai.pt"

# Load an image from the file system (assumed to be in the same folder as this script).
logo = Image.open(LOGO)

# Configure the page
st.set_page_config(page_title="micronuclAI", layout="centered")

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
st.image(logo, width="stretch")

# Section to explain how to use the app.
st.title("How to use this app", text_alignment="center")
col1, col2 = st.columns(2, vertical_alignment="center")
with col1:
    st.write(
        """
        This app shocases micronuclAI, it allows you to test our pretrained model with your own small test data.

        For larger datasets please refer to the CLI application of micronuclAI and try it locally on your machine or cluster.

        Check the CLI implementation at: https://github.com/SchapiroLabor/micronuclAI
        """
    )
with col2:
    with open(TEST_DATA, "rb") as f:
        st.download_button(
            "Download test data", f, file_name="testdata.tar.gz", width="stretch"
        )

# Define input files
col1, col2, col3 = st.columns(3, vertical_alignment="center")
with col1:
    nuclei_image = st.file_uploader(
        "Upload a nuclear staining file:",
        accept_multiple_files=False,
        key="nuclei_image",
    )

with col2:
    mask_image = st.file_uploader(
        "Upload a nuclear mask file:", accept_multiple_files=False, key="nuclei_mask"
    )
with col3:
    submit_button = st.button(
        "Run the script", key="submit_button_key", width="stretch"
    )


# Get the file paths from the uploaded file
# Example of how to test temp_maskimage.name = 'mask.tif'
if nuclei_image is not None:
    temp_nucimage = tempfile.NamedTemporaryFile(prefix="nuclei.", dir=".")
    temp_nucimage.write(nuclei_image.getbuffer())

if mask_image is not None:
    # here I call it micronuclAI because prediction2 creates multiple files based on this name so in the end I will have micronuclAI_predictions.csv and micronuclAI_summary.csv
    temp_maskimage = tempfile.NamedTemporaryFile(prefix="micronuclAI.", dir=".")
    temp_maskimage.write(mask_image.getbuffer())

################
## Run inference when user clicks run button

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None

if "chart_summary_df" not in st.session_state:
    st.session_state.chart_summary_df = None

if "micro_sum_df" not in st.session_state:
    st.session_state.micro_sum_df = None

if "predictions_csv" not in st.session_state:
    st.session_state.predictions_csv = None

if "summary_csv" not in st.session_state:
    st.session_state.summary_csv = None

placeholder = st.empty()

if submit_button:
    if nuclei_image is None or mask_image is None:
        st.warning(
            "Please upload both a nuclear staining file and a nuclear mask file."
        )
    else:
        with placeholder.container():
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(GIF)

        subprocess.run(
            [
                f"{sys.executable}",
                "prediction2.py",
                "-i",
                temp_nucimage.name,
                "-m",
                temp_maskimage.name,
                "-mod",
                MODEL,
                "-o",
                RESULTS,
            ]
        )

        placeholder.empty()

        pred_out = "micronuclAI_predictions.csv"
        sum_out = "micronuclAI_summary.csv"

        predictions = pd.read_csv(RESULTS / pred_out)
        micro_sum = pd.read_csv(RESULTS / sum_out)

        chart_summary = (
            predictions.groupby("micronuclei").size().reset_index(name="count")
        )
        chart_summary["micronuclei"] = chart_summary["micronuclei"].astype(str)

        st.session_state.predictions_df = predictions
        st.session_state.chart_summary_df = chart_summary
        st.session_state.micro_sum_df = micro_sum
        st.session_state.predictions_csv = predictions.to_csv(index=False).encode(
            "utf-8"
        )
        st.session_state.summary_csv = micro_sum.to_csv(index=False).encode("utf-8")
        st.session_state.results_ready = True

## Result visualization
if st.session_state.results_ready:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Micronuclei distribution", text_alignment="center")
        fig = px.bar(
            st.session_state.chart_summary_df,
            x="micronuclei",
            y="count",
            template="simple_white",
        )
        fig.update_traces(marker_color="rgb(0,119,182)")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Summary Table", text_alignment="center")
        st.dataframe(st.session_state.micro_sum_df)

    col1, col2 = st.columns(2, vertical_alignment="center")

    with col1:
        st.download_button(
            label="Download predictions",
            data=st.session_state.predictions_csv,
            file_name="micronuclAI_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    with col2:
        st.download_button(
            label="Download summary",
            data=st.session_state.summary_csv,
            file_name="micronuclAI_summary.csv",
            mime="text/csv",
            width="stretch",
        )

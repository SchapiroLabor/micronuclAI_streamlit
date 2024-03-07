import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Function to download a file.
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
    return href

# Load an image from the file system (assumed to be in the same folder as this script).
logo = Image.open('logo.png')

# Use the full page instead of a narrow central column
st.set_page_config(layout="centered")

# Display the logo at the top of the page.
st.image(logo, use_column_width=True)

# Section to explain how to use the app.
st.header("How to use")
st.write("This section will contain instructions on how to use the app.")

# Upload buttons for data.
st.header("Upload Data")
uploaded_file = st.file_uploader("Choose a file to upload")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write("Uploaded file.")

# Download test data button.
## st.markdown(get_binary_file_downloader_html('test_data.zip', 'Test Data'), unsafe_allow_html=True)

# Displaying plots and tables.
st.header("Data Visualization")
col1, col2 = st.beta_columns(2)
with col1:
    st.subheader("Distribution Plot")
    st.write("This plot shows the distribution of micronuclei.")

with col2:
    st.subheader("Summary Table")
    st.write("This table summarizes the data.")

# Scatterplot placeholder.
st.header("Scatterplot")
st.write("Scatterplot of x,y with 2 different colored nuclei (1=no micronuclei, 2=micronuclei)")

# Download zipped data package.
st.header("Download Data")
st.markdown(get_binary_file_downloader_html('data_package.zip', 'Data Package'), unsafe_allow_html=True)

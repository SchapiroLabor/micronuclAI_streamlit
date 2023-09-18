## Run streamlit app from within docker container
## To build docker container : docker build -t cin_streamlit .

# docker run -p 8501:8501 -it --rm cin_streamlit:latest streamlit run /app/streamlit_example.py --server.port=8501 --server.address=0.0.0.0
docker run -p 8501:8501 -it --rm \
-v "/Users/florian_wuennemann/1_Projects/Miguel_CIN_streamlit/results:/results" \
-v "/Users/florian_wuennemann/1_Projects/Miguel_CIN_streamlit/models:/models" \
cin_streamlit
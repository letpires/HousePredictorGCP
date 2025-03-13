# HousePredictorGCP

This project is designed to predict house prices using a machine learning model. The project consists of two main components: a Jupyter Notebook (`house_prediction.ipynb`) that runs on Vertex AI Workbench, and a Streamlit application (`app.py`) that provides a user-friendly interface for making predictions.

## Project Overview

The goal of this project is to demonstrate a complete machine learning workflow, from data extraction and model training to deploying the model in production using a Streamlit application. This approach simplifies the process of making predictions by providing an easy-to-use web interface.

## Prerequisites

- Python 3.7 or higher
- Google Cloud SDK
- Vertex AI Workbench
- Streamlit
- Pandas
- NumPy
- Joblib
- Google Cloud Storage
- Google OAuth2
- Google BigQuery

## Setup

### Google Cloud Setup

1. **Create a Google Cloud Project**: If you don't have a Google Cloud project, create one [here](https://console.cloud.google.com/).

2. **Enable APIs**: Enable the following APIs for your project:
   - Vertex AI API
   - Cloud Storage API
   - BigQuery API

3. **Create a Vertex AI Workbench**: Set up a Vertex AI Workbench instance to run the Jupyter Notebook.

4. **Create a Cloud Storage Bucket**: Create a bucket to store your model files.

5. **BigQuery Dataset**: Ensure you have a BigQuery dataset and table containing the house data. The table should be relational and include relevant features for house price prediction.

### Local Setup

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/HousePredictorGCP.git
    cd HousePredictorGCP
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Jupyter Notebook on Vertex AI Workbench

1. **Upload the Notebook**: Upload `house_prediction.ipynb` to your Vertex AI Workbench instance.

2. **Run the Notebook**: Open the notebook in Vertex AI Workbench and run all cells to train the model and save the pipeline to your Cloud Storage bucket. The notebook pulls data from a BigQuery table for training.

## Running the Streamlit Application

1. **Service Account Key**: Download the service account key JSON file from the Google Cloud Console and place it in the project directory. Ensure the service account has access to the Cloud Storage bucket and BigQuery dataset.

2. **Update the Path to the Service Account Key**: In `app.py`, update the path to the service account key JSON file:
    ```python
    credentials = service_account.Credentials.from_service_account_file(
        "/path/to/your/service-account-file.json"
    )
    ```

3. **Run the Streamlit App**:
    ```sh
    streamlit run app.py
    ```

4. **Access the App**: Open your web browser and go to `http://localhost:8501` to access the Streamlit application.

## Project Structure

- `app.py`: Streamlit application for house price prediction.
- `house_prediction.ipynb`: Jupyter Notebook for training the model on Vertex AI Workbench.
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: Project documentation.

## Usage

### Streamlit Application

1. Open the Streamlit application in your web browser.
2. Enter the property data in the form provided.
3. Click the "Predict Price" button to get the estimated price of the property.

### Jupyter Notebook

1. Open the `house_prediction.ipynb` notebook in Vertex AI Workbench.
2. Run all cells to train the model and save the pipeline to your Cloud Storage bucket. The notebook pulls data from a BigQuery table for training.

## Data Source

The data used for training the model is pulled from a BigQuery table. Ensure that the table is relational and includes relevant features such as:
- Number of bedrooms
- Number of bathrooms
- Living area size
- Lot area size
- Number of floors
- Waterfront indicator
- View rating
- Property grade
- Renovation status
- Basement size
- Property condition

## Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [Google Cloud](https://cloud.google.com/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)
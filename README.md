# flight-delay-prediction-svm
Project Overview

This project aims to predict flight arrival delays using Support Vector Machines (SVM). The model utilizes phase space reconstruction techniques and upstream flight delay data to analyze and predict patterns in flight arrival delays. The project leverages time series analysis and upstream departure delays as input features to develop an effective prediction model.

Features

Input Data: The dataset includes information such as upstream delays and departure times.

Feature Engineering: Phase space reconstruction theory is applied to identify chaotic characteristics in the flight delay data.

Prediction Model: The SVM model uses upstream flight delays and time series data to predict flight arrival delays.

Tools & Technologies: Python, scikit-learn, pandas, matplotlib.

Installation and Setup

To run the project, follow these steps:

Clone the repository:

git clone https://github.com/your-username/flight-delay-prediction-svm.git

Navigate to the project directory:

cd flight-delay-prediction-svm

Install the required dependencies:

pip install -r requirements.txt

Make sure you have a dataset named flight_data.csv in the root directory, or modify the code to point to your data file.

Running the Project

To run the prediction model:

python src/flight_delay_prediction.py

The model will train on the flight data and output evaluation metrics. Additionally, it will store the results, including:

Evaluation Metrics: Stored in results/evaluation_metrics.txt.

Prediction Plot: Stored as results/actual_vs_predicted.png.

Project Structure

src/: Contains the main Python script (flight_delay_prediction.py).

data/: Placeholder for the dataset file (flight_data.csv).

results/: Stores the evaluation metrics and prediction plots.

README.md: Project documentation.

requirements.txt: Lists all dependencies required for running the project.

Results

Mean Squared Error (MSE) and R-squared (R²) metrics are provided to evaluate the model's performance.

Actual vs. Predicted Plot: A scatter plot is generated to visually compare actual and predicted delays.


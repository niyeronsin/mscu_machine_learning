# Heart Disease Classification Project

## Overview

This Jupyter Notebook contains the analysis and modeling for a heart disease classification project. The goal is to develop a supervised machine learning model that can accurately predict the presence of heart disease in patients based on a set of clinical features.

The project utilizes the Cleveland dataset from the UCI Machine Learning Repository, which consists of 13 features and a target variable indicating the presence and severity of heart disease.

## Project Structure

- **`supervised_learning_final_project.ipynb`**: The main Jupyter Notebook containing the data analysis, model building, and evaluation.

## Dataset

- **Source**: UCI Machine Learning Repository - Heart Disease Dataset (Cleveland dataset)
- **Features**: 13 clinical features (detailed descriptions are in the notebook).
- **Target Variable**: `num` (0 = No heart disease, 1-4 = Presence of heart disease with varying severity).
- **Number of Samples**: 303

## Libraries Used

- `ucimlrepo`: For fetching the dataset from the UCI repository.
- `pandas`: For data manipulation and analysis.
- `matplotlib.pyplot`: For data visualization.
- `seaborn`: For enhanced data visualization.
- `scikit-learn`: For machine learning models and evaluation metrics.

## Analysis and Modeling

The notebook includes the following steps:

1.  **Data Loading and Exploration**: Fetching the dataset, exploring its structure, and understanding the features.
2.  **Exploratory Data Analysis (EDA)**: Visualizing the data to identify patterns and relationships between features and the target variable.
3.  **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numerical features.
4.  **Model Building**: Training and evaluating machine learning models (Logistic Regression and Random Forest).
5.  **Model Evaluation**: Assessing model performance using metrics such as accuracy, precision, recall, F1-score, and AUC.
6.  **Model Comparison and Conclusion**: Comparing the performance of the models and selecting the best model based on the project goals.

## Key Findings

-   Logistic Regression and Random Forest models were trained and evaluated.
-   Logistic Regression performed slightly better in terms of recall, which is crucial for medical diagnoses where minimizing false negatives is essential.
-   Random Forest provided a more balanced performance between precision and recall.
-   The choice of the model depends on the specific priorities of the application. For heart disease detection, Logistic Regression is preferred due to its higher recall.

## How to Run

1.  Ensure you have Python 3 and the required libraries installed.
2.  Clone this repository.
3.  Navigate to the repository directory.
4.  Run the Jupyter Notebook `supervised_learning_final_project.ipynb`.

## Author

-   Olaniyi Nafiu

## Contact

For any questions or feedback, please feel free to reach out.
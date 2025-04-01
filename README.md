# Heart-Disease-Prediction-using-Stacked-Ensemble-Learning-Models
The primary aim of this project is to develop a reliable and efficient system that accurately predicts the risk of heart disease, thereby enabling healthcare professionals to make informed decisions for early diagnosis and intervention.

# Abstract
Cardiovascular diseases (CVDs) are a leading cause of mortality globally, underscoring the
necessity for effective predictive models to facilitate early diagnosis and intervention. This
study introduces a machine learning-based system designed to predict heart disease by
employing five classification algorithms: Decision Tree, Random Forest, Support Vector
Classifier (SVC), XGBoost, and Logistic Regression. The model is trained on a dataset
comprising 3,215 patient records, with 1,735 positive and 1,480 negative cases, aiming to
enhance predictive accuracy through the integration of multiple algorithms. The data
preprocessing pipeline encompasses handling missing values, feature scaling, and the
application of the Synthetic Minority Over-sampling Technique (SMOTE) to address class
imbalance. Feature correlation analysis and hyperparameter tuning are conducted to optimize
model performance. Following the training phase, all five models are combined into a stacked
ensemble to leverage their collective strengths, resulting in a more robust and generalizable
predictive model. The final stacked model demonstrates high predictive performance,
achieving an accuracy of 95.33%, precision of 93.53%, recall of 98.30%, F1 score of 95.86%,
and an ROC-AUC score of 96.65%. Evaluation on a separate dataset yields a consistent
accuracy of 96.88%, indicating the model's reliability. To enhance accessibility, the system is
deployed as a web application, allowing users to input health data and receive real-time heart
disease risk predictions. This tool offers significant potential for healthcare professionals and
patients, providing an efficient method for early heart disease detection and risk assessment.

# Hardware Requirement
* Processor: Intel Core i3 or equivalent (minimum 2.0 GHz).
* RAM: 8 GB (minimum) to support smooth operation, especially during data
processing and model training.
* Hard Disk: 500 GB (minimum) to store the dataset, model files, and related project
resources.
* Display: A screen with a minimum resolution of 1366x768 for ease of visualization
during model training, testing, and analysis.
* Graphics Card: A dedicated or integrated graphics card with support for modern
software, although ts project primarily relies on CPU-based processing.

# Software Requirement
* IDE: PyCharm (Preferred Integrated Development Environment for Python
development).
* Operating System: Windows, Linux, or macOS (depending on user preference).
* Python Version: Python 3.x (recommended version 3.6 or higher).

# Libraries and Packages:
* Scikit-learn: For implementing machine learning models like Decision Tree, Random
Forest, Logistic Regression, and SVC.
* XGBoost: For the XGBoost classifier, which is used to enhance model accuracy
through gradient boosting.
* Pandas: For data manipulation, cleaning, and analysis.
* NumPy: For numerical computations and handling arrays.
* Matplotlib and seaborn: For data visualization (graphs, charts, heatmaps).
  Imbalanced-learn (SMOTE): For handling class imbalance in the dataset using the
Synthetic Minority Over-sampling Technique (SMOTE).
* Joblib: For saving and loading trained machine learning models.
* Flask: For deploying the model as a web application.

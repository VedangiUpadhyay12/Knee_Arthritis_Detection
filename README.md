# Knee_Arthritis_Detection
This project aims to automatically detect and predict the severity of knee osteoarthritis using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). By leveraging medical imaging and deep learning, it supports early diagnosis and tracking of disease progression to assist radiologists and clinicians in decision-making.
📌 Features
🔍 Multi-class classification of knee arthritis severity (Grades 0–4) from X-ray images

🎯 Built with a custom CNN model trained on labeled datasets

🔥 Integrated Grad-CAM visualization to interpret model predictions

📈 Predicts arthritis progression trend using RNN (LSTM/GRU)

📂 Export predictions from auto_test to CSV for reporting

🌐 Interactive Web UI built with Streamlit or Flask for easy user interaction

🧠 Technologies Used
TensorFlow / Keras

OpenCV & Matplotlib

Grad-CAM (Model Explainability)

RNN (LSTM/GRU) for progression prediction

Streamlit (or Flask) for Web Interface

Google Colab for training with Google Drive integration

📁 Dataset Structure
The dataset contains four main folders:

train/, val/, test/, auto_test/
Each of these contains subfolders:

0/, 1/, 2/, 3/, 4/ → Corresponding to arthritis severity grades

🚀 How to Run
Mount Google Drive in Colab

Load dataset from Drive to /content/

Train CNN on the train/ and val/ data

Run Grad-CAM on test images

Use RNN to analyze arthritis progression

Launch the Streamlit Web UI to interact with the model


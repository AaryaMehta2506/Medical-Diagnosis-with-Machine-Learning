AI/ML Advance Project
# Medical Diagnosis with Machine Learning

A machine learning web application that predicts medical conditions based on patient details such as age, gender, blood type, billing amount, admission type, test results, and length of hospital stay. The model uses classification techniques to identify likely diagnoses and displays prediction confidence levels in a Streamlit interface.

## Features
- Predicts medical conditions using demographic and clinical details  
- Interactive web interface built with Streamlit  
- Displays prediction confidence for top probable conditions  
- Scalable for integration with hospital data systems

## Dataset 
Link : https://www.kaggle.com/datasets/prasad22/healthcare-dataset

## Project Structure
```
Medical-Diagnosis-ML/
│
├── app.py                     # Streamlit web application
├── train_model.py             # Model training script
├── healthcare_dataset.csv     # Dataset used for model training
├── medical_diagnosis_rf.joblib  # Trained Random Forest model
├── medical_scaler.joblib      # Scaler used for numeric features
├── medical_label_encoder.joblib  # Label encoder for target variable
└── README.md                  # Project documentation
```

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/your-username/Medical-Diagnosis-ML.git
cd Medical-Diagnosis-ML

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn streamlit joblib matplotlib

# Optional: Install OpenCV and imgaug if using image data
pip install opencv-python imgaug
```

## Run the Application
```bash
# Train the model (if not pre-trained)
python train_model.py

# Launch the Streamlit app
streamlit run app.py
```

## How It Works
1. The training script loads the healthcare dataset, performs preprocessing, and trains a classification model.  
2. The trained model, scaler, and label encoder are saved using joblib.  
3. The Streamlit app loads these artifacts, accepts user inputs, scales and encodes features, and makes predictions.  
4. The predicted condition and confidence levels are displayed interactively.

## Technologies Used
- Python  
- Scikit-learn  
- Pandas, NumPy  
- Streamlit  
- Joblib  

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)



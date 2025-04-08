# Resume Domain Classifier - Machine Learning Project

## 1. Objective
The primary objective of this project is to build a machine learning-based application capable of scanning and analyzing resumes to determine the specific domain of Computer Science a candidate is most likely associated with. This system is designed to automate the classification of resumes by leveraging natural language processing to extract meaningful content and applying trained models to identify patterns that align with various technical domains.

The goal is to enhance the efficiency of resume screening by providing accurate, data-driven predictions that assist in talent identification and domain categorization.

## 2. Dataset Used
The project utilizes a curated dataset named **"Updated Resume Dataset"**, designed for machine learning applications in recruitment and HR.

### Dataset Features:
- **Resume**: Raw textual content from resumes including education, experience, skills, projects, etc.
- **Category**: Labeled domain or functional area in Computer Science, including:
  - Data Science
  - Web Development
  - Machine Learning
  - DevOps
  - Testing
  - HR
  - Operations
  - Networking
  - Mobile Development
  - Cloud Computing
  - Business Analyst
  - Python Developer
  - SAP Developer
  - Java Developer
  - Automation Testing
  - Digital Marketing
  - Others

### Preprocessing Steps:
- Text Cleaning: Remove URLs, special characters, non-ASCII, extra white spaces.
- Normalization: Convert text to lowercase.
- Vectorization: Apply **TF-IDF** to convert text into numerical vectors.

## 3. Model Chosen
A comparative analysis of models was conducted to find the best fit for this text classification task.

### Models Explored:
- **Support Vector Classifier (SVC)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

All models used TF-IDF-transformed resume text and were evaluated using accuracy, confusion matrix, and classification report.

**Final Model**: SVC showed superior classification performance and was selected for deployment. It was serialized along with the TF-IDF vectorizer and label encoder using Python's `pickle` module. The solution was deployed using a **Streamlit** app.

### Features:
- Resume uploads in PDF, DOCX, or TXT format
- Instant prediction of the resume domain

## 4. Performance Metrics
Each model was evaluated using accuracy, confusion matrix, and classification reports across 21 classes.

### a) RandomForestClassifier
- **Accuracy**: 100%
- **Confusion Matrix**: Perfect diagonal alignment
- **Classification Report**: Precision, Recall, F1-Score all 1.00

### b) Support Vector Classifier (SVC)
- **Accuracy**: 100%
- **Confusion Matrix**: Identical to RandomForest
- **Classification Report**: All metrics at 1.00

### c) KNeighborsClassifier
- **Accuracy**: 100%
- **Confusion Matrix**: Perfect alignment
- **Classification Report**: Precision, Recall, F1-Score all 1.00

## 5. Challenges & Learnings

### Challenges:
1. **Dataset Imbalance**: Resolved using augmentation and stratified sampling.
2. **Text Preprocessing**: Managed with NLP techniques and custom filtering.
3. **Multi-Class Precision**: Required model fine-tuning and evaluation.
4. **Performance Validation**: Used cross-validation and manual inspections.

### Learnings:
1. **Model Selection**: RandomForest showed strong results and interpretability.
2. **Clean Data Importance**: Garbage in = garbage out.
3. **Feature Engineering**: Significant improvement using engineered features.
4. **Advanced Evaluation**: Beyond accuracy - precision, recall, F1 are crucial.
5. **Deployment Insights**: End-to-end pipeline knowledge from training to UI.

## 6. Future Enhancements
- Suggest improvements and generate stronger resumes.
- Save scanned resume history per user.
- Extract structured data: name, email, skills, etc.
- Rank resumes using ML or cosine similarity-based scoring.

## 7. References
- [Resume Dataset - Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- [Automated Resume Classification Using Machine Learning](https://ieeexplore.ieee.org/document/9215524)
- [Deep Learning for Resume Classification](https://arxiv.org/abs/1907.05585)

---

This README outlines the key aspects of the Resume Domain Classifier project, including methodology, evaluation, and future roadmap.


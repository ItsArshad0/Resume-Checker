import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:/Users/arsha/OneDrive/Desktop/C++ DSA/NLP/UpdatedResumeDataSet.csv")
print(df)
print(df['Category'].value_counts())

print(df.shape)
print(plt.figure(figsize=(15,5)))

sns.countplot(df['Category'])
plt.xticks(rotation=90)
print(plt.show())

import re

def CleanResume(txt):
    cleanTxt = re.sub(r"http\S+", '', txt)  # Remove URLs
    cleanTxt = re.sub(r'\bRT\b|\bCC\b', " ", cleanTxt)  # Remove 'RT' and 'CC' as standalone words
    cleanTxt = re.sub(r"@\S+", '', cleanTxt)  # Remove mentions
    cleanTxt = re.sub(r"#\S+", '', cleanTxt)  # Remove hashtags
    cleanTxt = re.sub(r'[^\w\s]', ' ', cleanTxt)  # Remove special characters but keep words and spaces
    cleanTxt = re.sub(r'[^\x00-\x7F]+', ' ', cleanTxt)  # Remove non-ASCII characters
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()  # Remove extra spaces

    return cleanTxt



print(CleanResume("welcome #### ^%$$$@@!! to http://randomizer welcome to it and my @gmail.com"))

print(df['Resume'].apply(lambda x: CleanResume(x)))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df['Category'])
df['Category']=le.transform(df['Category'])
print(df['Category'])

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')

tfidf.fit(df['Resume'])
TextTrans=tfidf.transform(df['Resume'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(TextTrans,df['Category'],test_size=0.2,random_state=42)
print(X_test.shape)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure that X_train and X_test are dense if they are sparse
X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

# 1. Train KNeighborsClassifier
knn_model = OneVsRestClassifier(KNeighborsClassifier())
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("\nKNeighborsClassifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_knn)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_knn)}")

svc_model = OneVsRestClassifier(SVC())
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
print("\nSVC Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svc):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svc)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_svc)}")

rf_model = OneVsRestClassifier(RandomForestClassifier())
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandomForestClassifier Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")

import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(svc_model, open('clf.pkl', 'wb'))
pickle.dump(le, open("encoder.pkl",'wb'))

def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = CleanResume(input_resume) 

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])
    
    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0] 

myresume=""" **John Doe**  
Email: johndoe@example.com | Phone: (123) 456-7890 | LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe  


### **Professional Summary**  
Results-driven Data Scientist with 5+ years of experience specializing in **Natural Language Processing (NLP)** and **Machine Learning (ML)**. Expertise in **text analytics, sentiment analysis, named entity recognition (NER), topic modeling, and chatbot development**. Passionate about leveraging AI to drive business insights and improve user experiences.

---

### **Skills**  
- **Programming:** Python, R, SQL  
- **NLP Frameworks:** spaCy, NLTK, Transformers (Hugging Face), Gensim  
- **Machine Learning:** Scikit-learn, TensorFlow, PyTorch  
- **Data Processing:** Pandas, NumPy, OpenAI GPT  
- **Cloud & Deployment:** AWS, GCP, Docker, Kubernetes  
- **Databases:** PostgreSQL, MongoDB, Elasticsearch  
- **Version Control & Tools:** Git, Jupyter Notebooks, MLflow  

---

### **Work Experience**  
#### **Senior Data Scientist**  
XYZ Tech Solutions | Jan 2021 – Present  
- Designed and implemented an **NLP-powered resume screening tool**, reducing hiring time by 40%.
- Developed a **custom Named Entity Recognition (NER) model** to extract key information from resumes.
- Built **sentiment analysis and topic modeling** solutions to analyze customer feedback.
- Led a team in **fine-tuning transformer models** (BERT, RoBERTa) for text classification tasks.

#### **Data Scientist**  
ABC Analytics | Jun 2018 – Dec 2020  
- Built an **automated resume parsing system** using **spaCy and NLTK**, improving HR efficiency by 35%.
- Developed a **text summarization model** for processing large volumes of legal documents.
- Implemented **semantic search** using Elasticsearch for document retrieval systems.

---

### **Education**  
**Master of Science in Data Science**  
University of XYZ | 2016 – 2018  

**Bachelor of Computer Science**  
ABC University | 2012 – 2016  

---

### **Certifications**  
- **Deep Learning Specialization** – Coursera (Andrew Ng)  
- **AWS Certified Machine Learning – Specialty**  
- **Natural Language Processing with Python** – Udemy  

---

### **Projects**  
- **Resume Screening AI:** Developed an NLP-based system to extract candidate skills and rank applicants based on job descriptions.
- **Chatbot for HR:** Created an interactive chatbot using **GPT-3** to automate HR-related queries.
- **Legal Document Analyzer:** Built an **AI-powered document summarization tool** for law firms.

---

### **Publications & Research**  
Improving Resume Screening with Deep Learning– Published in XYZ Journal, 2022  
Enhancing Semantic Search with Transformers  IEEE Conference Paper, 2021  
### **Additional Information**  
Open to relocation and remote work  
Actively contributing to NLP open-source projects  
"""
print(pred(myresume))


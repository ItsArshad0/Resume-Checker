import streamlit as st
import pickle
import docx  # Extract text from Word file
import docx as docx_lib
import PyPDF2  # Extract text from PDF
import re

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Example file name, adjust as needed


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Check if text is extracted
            text += page_text + "\n"
    return text if text else "No text could be extracted from this PDF."


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text if text else "No text found in the DOCX file."


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')  # Fallback encoding
    return text if text else "No text found in the TXT file."


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]  # Return the category name


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        resume_text = handle_file_upload(uploaded_file)

        if resume_text.startswith("Error"):
            st.error(resume_text)  # Show error message
        else:
            st.success("Successfully extracted text from the resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")


if __name__ == "__main__":
    main()
    

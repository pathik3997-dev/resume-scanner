import streamlit as st
import re
import string
import PyPDF2
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Keyword Scanner", layout="centered")

st.title("📄 AI Resume Keyword Scanner (PDF Upload + ATS Tips)")

# Upload PDF Resume
uploaded_file = st.file_uploader("📎 Upload Your Resume (PDF)", type=["pdf"])

# Job Description text
job_desc_text = st.text_area("📝 Paste the Job Description here", height=200)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

if st.button("Scan Resume"):
    if not uploaded_file or not job_desc_text:
        st.warning("Please upload a resume and paste the job description.")
    else:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(uploaded_file)

        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_desc_text)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        st.success(f"✅ Resume Match Score: **{round(similarity * 100, 2)}%**")

        jd_keywords = set(jd_clean.split())
        resume_keywords = set(resume_clean.split())

        matched = jd_keywords.intersection(resume_keywords)
        missing = jd_keywords.difference(resume_keywords)

        st.write("### ✅ Matched Keywords:")
        st.write(", ".join(matched) if matched else "None")

        st.write("### ❌ Missing Keywords:")
        st.write(", ".join(missing) if missing else "None")

        if missing:
            suggestions = [
                f"Consider adding keywords like **{word}** to highlight relevant experience."
                for word in list(missing)[:5]
            ]
            st.write("### 💡 Suggested Improvements for ATS:")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")

        # ✅ ATS Formatting Tips
        st.write("### 📌 ATS-Friendly Formatting Checklist")
        st.markdown("""
        - ✅ Use standard headings like `Experience`, `Skills`, `Education`
        - ✅ Save the resume as a plain PDF (not scanned image)
        - ❌ Avoid using tables, columns, or text boxes
        - ❌ Don’t include icons, logos, or charts
        - ✅ Use simple fonts like Arial, Calibri, or Times New Roman
        - ✅ Use plain bullet points (● or -)
        - ❌ No colors or graphics (especially for ATS submission)
        """)

        # Resume Summary (Optional Display)
        st.write("### 📄 Resume Content Preview:")
        st.text_area("Extracted Resume Text", resume_text, height=200)

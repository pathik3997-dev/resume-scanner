import streamlit as st
import PyPDF2
import io
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from fpdf import FPDF
from docx import Document

st.set_page_config(page_title="ATS-Free Resume Scanner", layout="wide")
st.title("📄 ATS-Free Resume Scanner")

# -------------------- Utility Functions --------------------
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def rule_based_suggestions_v2(resume_text, job_keywords):
    suggestions = []
    resume_lower = resume_text.lower()
    
    if len(resume_text.split("\n")) < 10:
        suggestions.append(("Improve spacing and use section headings", False))
    if not re.search(r"\bexperience\b", resume_lower):
        suggestions.append(("Add an 'Experience' section", False))
    if not re.search(r"\beducation\b", resume_lower):
        suggestions.append(("Include 'Education' details", False))
    if not re.search(r"\bprojects?\b", resume_lower):
        suggestions.append(("Include projects to showcase practical skills", False))

    if any(word in resume_lower for word in ["responsible", "duties", "assigned"]):
        suggestions.append(("Use result-oriented phrases instead of duties", False))

    if not re.search(r"\d+%", resume_text) and not re.search(r"\$\d+", resume_text):
        suggestions.append(("Quantify results with metrics", False))

    if not any(word in resume_lower for word in ["led", "built", "created", "improved", "solved", "increased", "initiated"]):
        suggestions.append(("Use strong action verbs", False))

    if len(resume_text) > 3000:
        suggestions.append(("Limit resume to 1–2 pages", False))
    if len(re.findall(r"[A-Z]{5,}", resume_text)) > 10:
        suggestions.append(("Reduce excessive capitalization", False))

    if not job_keywords & set(resume_lower.split()):
        suggestions.append(("Add keywords relevant to the job", False))

    if not suggestions:
        suggestions.append(("🎉 Your resume looks great!", True))

    return suggestions

def export_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    lines = text.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin-1')

def export_to_docx(text):
    doc = Document()
    doc.add_heading("Resume", 0)
    for line in text.split('\n'):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

# -------------------- App Layout --------------------
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description (Optional for Keyword Suggestions)")

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    clean_resume = clean_text(resume_text)

    # Word Cloud
    st.subheader("📊 Resume Word Cloud")
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(clean_resume)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Bar Chart for Keywords
    st.subheader("📈 Top Keywords")
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform([clean_resume])
    keyword_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
    df_keywords = pd.DataFrame(list(keyword_freq.items()), columns=['Keyword', 'Frequency'])
    st.bar_chart(df_keywords.set_index('Keyword'))

    # Suggestions Checklist UI
    st.subheader("✅ Suggestions Checklist")
    job_keywords = set(job_description.lower().split()) if job_description else set()
    suggestions = rule_based_suggestions_v2(resume_text, job_keywords)

    for tip, done in suggestions:
        st.checkbox(tip, value=done, disabled=True)

    # Before/After Sample Example
    st.subheader("✏️ Before vs After Tip Example")
    st.markdown("**Before:** Responsible for managing a team of developers.\n\n**After:** Led a team of 5 developers to deliver 3 successful projects ahead of schedule.")

    # Resume Re-download
    st.subheader("📥 Download Your Resume")
    with open(uploaded_file.name, "rb") as f:
        resume_bytes = f.read()
    st.download_button("Download Original Resume", data=resume_bytes, file_name="your_resume.pdf")

    col1, col2 = st.columns(2)
    with col1:
        pdf_bytes = export_to_pdf(resume_text)
        st.download_button("⬇ Export as PDF (Text Format)", data=pdf_bytes, file_name="resume_export.pdf")
    with col2:
        docx_bytes = export_to_docx(resume_text)
        st.download_button("⬇ Export as DOCX (Text Format)", data=docx_bytes, file_name="resume_export.docx")

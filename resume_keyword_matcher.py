import streamlit as st
import PyPDF2
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

st.set_page_config(page_title="ATS Resume Scanner", layout="wide")
st.title("📄 ATS Resume Scanner (No GPT)")

# Upload PDF Resume
resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

# Input Job Description
job_desc = st.text_area("Paste Job Description Here")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def extract_sections(text):
    sections = {
        "Experience": "",
        "Skills": "",
        "Education": "",
        "Projects": ""
    }
    current_section = None
    lines = text.split("\n")
    for line in lines:
        line = line.strip().lower()
        if "experience" in line:
            current_section = "Experience"
        elif "skills" in line:
            current_section = "Skills"
        elif "education" in line:
            current_section = "Education"
        elif "project" in line:
            current_section = "Projects"
        elif current_section:
            sections[current_section] += line + " "
    return sections

if resume_file and job_desc:
    with st.spinner("Extracting text from resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        resume_clean = clean_text(resume_text)
        job_desc_clean = clean_text(job_desc)

        vectorizer = TfidfVectorizer(stop_words="english")
        vectorizer.fit([job_desc_clean])
        job_keywords = set(vectorizer.get_feature_names_out())
        resume_words = set(resume_clean.split())

        missing_keywords = job_keywords - resume_words
        match_count = len(job_keywords) - len(missing_keywords)
        match_percent = round((match_count / len(job_keywords)) * 100, 2)

        # Show ATS Score
        st.subheader("✅ ATS Keyword Match Score")
        st.progress(int(match_percent))
        st.write(f"**Match: {match_count} / {len(job_keywords)} keywords ({match_percent}%)**")

        # Show Missing Keywords
        st.subheader("🚫 Missing Keywords")
        st.write(", ".join(list(missing_keywords)[:30]))

        # Word Cloud
        st.subheader("📊 Resume Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_clean)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        # Section-Wise Breakdown
        st.subheader("📁 Section-Wise Breakdown")
        sections = extract_sections(resume_text)
        for sec, content in sections.items():
            if content.strip():
                sec_keywords = set(content.lower().split())
                found = sec_keywords & job_keywords
                st.markdown(f"**{sec}** – Keywords matched: {len(found)}")
                st.caption(", ".join(found) if found else "No keywords matched.")

else:
    st.info("Please upload a PDF resume and paste a job description to start.")

import streamlit as st
import pdfplumber
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import re

st.set_page_config(page_title="ATS Resume Scanner", layout="wide")

st.title("📄 ATS-Friendly Resume Scanner")
st.caption("Upload your resume and a job description to get a match score and missing keyword insights!")

# --- Upload resume
uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type=["pdf"])

# --- Paste job description
job_desc = st.text_area("📝 Paste the Job Description Here")

# --- Extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# --- Clean text
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.lower()

# --- Extract sections from resume (basic rule-based)
def extract_sections(text):
    sections = {
        "Experience": "",
        "Education": "",
        "Skills": "",
        "Projects": ""
    }
    current_section = None
    for line in text.split("\n"):
        line_clean = line.strip().lower()
        if "experience" in line_clean:
            current_section = "Experience"
        elif "education" in line_clean:
            current_section = "Education"
        elif "skill" in line_clean:
            current_section = "Skills"
        elif "project" in line_clean:
            current_section = "Projects"
        elif current_section:
            sections[current_section] += line + " "
    return sections

# --- Match keywords
def keyword_match(resume_text, job_desc):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_desc).split())
    common = resume_words & job_words
    missing = job_words - resume_words
    match_percent = round(len(common) / len(job_words) * 100, 2) if job_words else 0
    return match_percent, common, missing, job_words

# --- Main App Logic
if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    clean_resume = clean_text(resume_text)
    match_percent, common, missing, job_keywords = keyword_match(resume_text, job_desc)

    st.subheader("✅ Match Score")
    st.markdown(f"**{match_percent}%** of the job description keywords are present in your resume.")

    st.subheader("❌ Missing Keywords")
    if missing:
        st.warning(", ".join(list(missing)[:15]) + ("..." if len(missing) > 15 else ""))
    else:
        st.success("Your resume contains all the keywords!")

    # Word Cloud
    st.subheader("📊 Resume Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_resume)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # --- Section-Wise Keyword Match Chart
    st.subheader("📁 Section-Wise Keyword Match")

    sections = extract_sections(resume_text)
    section_names = []
    matched_counts = []
    matched_keywords = {}

    for sec, content in sections.items():
        if content.strip():
            sec_keywords = set(content.lower().split())
            found = sec_keywords & job_keywords
            section_names.append(sec)
            matched_counts.append(len(found))
            matched_keywords[sec] = found

    # Bar Chart
    fig2, ax2 = plt.subplots()
    ax2.barh(section_names, matched_counts, color='skyblue')
    ax2.set_xlabel('Number of Matching Keywords')
    ax2.set_title('Keyword Match by Resume Section')
    st.pyplot(fig2)

    # Section-wise text details
    for sec in section_names:
        st.markdown(f"**{sec}** – Keywords matched: {len(matched_keywords[sec])}")
        st.caption(", ".join(matched_keywords[sec]) if matched_keywords[sec] else "No keywords matched.")

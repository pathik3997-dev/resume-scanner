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

# --- Page Config ---
st.set_page_config(page_title="ATS-Free Resume Scanner", layout="wide")
st.title("📄 ATS-Free Resume Scanner")

# -------------------- Utility Functions --------------------

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def clean_text(text):
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_sections(text):
    sections = {"Experience": "", "Education": "", "Skills": "", "Projects": ""}
    current = None
    for line in text.split("\n"):
        l = line.strip().lower()
        if "experience" in l:
            current = "Experience"
        elif "education" in l:
            current = "Education"
        elif "skill" in l:
            current = "Skills"
        elif "project" in l:
            current = "Projects"
        elif current:
            sections[current] += line + "\n"
    return sections


def keyword_match(resume_text, job_desc):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_desc).split())
    common = resume_words & job_words
    missing = job_words - resume_words
    match_percent = round(len(common) / len(job_words) * 100, 2) if job_words else 0
    return match_percent, common, missing, job_words


def rule_based_suggestions_v2(resume_text, job_keywords):
    tips = []
    low = resume_text.lower()
    if len(resume_text.split("\n")) < 10:
        tips.append(("Improve spacing and add clear section headings", False))
    if "experience" not in low:
        tips.append(("Add an 'Experience' section", False))
    if "education" not in low:
        tips.append(("Include an 'Education' section", False))
    if "skill" not in low:
        tips.append(("Add a 'Skills' list with relevant tools", False))
    if "project" not in low:
        tips.append(("Include a 'Projects' section to showcase work", False))
    if not re.search(r"\d+%", resume_text) and not re.search(r"\$\d+", resume_text):
        tips.append(("Quantify achievements with numbers or metrics", False))
    if not any(w in low for w in ["led", "built", "created", "improved"]):
        tips.append(("Use action verbs like 'Led', 'Built', 'Improved'", False))
    if len(resume_text) > 3000:
        tips.append(("Limit resume to 1–2 pages for readability", False))
    if not job_keywords & set(low.split()):
        tips.append(("Add keywords from the job description", False))
    if not tips:
        tips.append(("🎉 Your resume looks strong and well-structured!", True))
    return tips

# -------------------- Export Functions --------------------

def export_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    # Write each line, stripping unsupported chars
    for line in text.split("\n"):
        safe_line = line.encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 10, safe_line)
    return pdf.output(dest='S').encode('latin-1')


def export_to_docx(text):
    doc = Document()
    doc.add_heading('Resume', 0)
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

# -------------------- Main App --------------------

uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste the Job Description (optional)")

if uploaded_file:
    resume_bytes = uploaded_file.read()
    resume_text = extract_text_from_pdf(io.BytesIO(resume_bytes))
    clean_resume = clean_text(resume_text)

    match_percent, matched, missing, job_keywords = keyword_match(resume_text, job_description)
    sections = extract_sections(resume_text)

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📁 Sections", "📥 Export & Tips"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("✅ Match Score")
            st.markdown(f"<h2 style='color:green'>{match_percent}%</h2>", unsafe_allow_html=True)
            st.subheader("❌ Missing Keywords")
            if missing:
                st.warning(", ".join(list(missing)[:20]) + ("..." if len(missing) > 20 else ""))
            else:
                st.success("All job keywords found!")

        with col2:
            st.subheader("☁️ Resume Word Cloud")
            wc = WordCloud(width=800, height=300, background_color='white', colormap='viridis', max_words=100).generate(clean_resume)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            st.pyplot(fig)

    with tab2:
        st.subheader("📊 Top Keywords")
        vec = CountVectorizer(stop_words='english', max_features=10)
        freqs = vec.fit_transform([clean_resume]).toarray()[0]
        df = pd.DataFrame({'Keyword': vec.get_feature_names_out(), 'Frequency': freqs})
        df = df.sort_values(by='Frequency', ascending=False)
        st.bar_chart(df.set_index('Keyword'))

        st.subheader("📁 Section-Wise Matches")
        names, counts = [], []
        sec_matches = {}
        for sec, cont in sections.items():
            if cont.strip():
                words = set(clean_text(cont).split())
                found = words & job_keywords
                names.append(sec); counts.append(len(found)); sec_matches[sec] = found
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        bars = ax2.barh(names, counts, color='#4CAF50')
        ax2.bar_label(bars); ax2.set_xlabel('Matches'); ax2.set_title('Section Matches')
        st.pyplot(fig2)
        for sec, kw in sec_matches.items():
            st.markdown(f"**{sec}**: {len(kw)} keywords"); st.caption(', '.join(kw) if kw else 'None')

    with tab3:
        st.subheader("📥 Download Original Resume")
        st.download_button("Download PDF", data=resume_bytes, file_name="your_resume.pdf", mime="application/pdf")

        st.subheader("⬇ Export as PDF (Text)")
        pdf_out = export_to_pdf(resume_text)
        st.download_button("Export PDF", data=pdf_out, file_name="resume_export.pdf", mime="application/pdf")

        st.subheader("⬇ Export as DOCX (Text)")
        docx_out = export_to_docx(resume_text)
        st.download_button("Export DOCX", data=docx_out, file_name="resume_export.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

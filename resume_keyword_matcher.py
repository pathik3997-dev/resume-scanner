import streamlit as st
import pdfplumber
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import io
import base64

st.set_page_config(page_title="ATS Resume Scanner", layout="wide")

st.title("📄 ATS-Friendly Resume Scanner")

# Upload + Job Description
uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📝 Paste the Job Description Here")

# Extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.lower()

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

def keyword_match(resume_text, job_desc):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_desc).split())
    common = resume_words & job_words
    missing = job_words - resume_words
    match_percent = round(len(common) / len(job_words) * 100, 2) if job_words else 0
    return match_percent, common, missing, job_words

def generate_txt_report(score, missing, matched, section_keywords):
    buffer = io.StringIO()
    buffer.write("ATS Resume Scanner Report\n")
    buffer.write("==========================\n\n")
    buffer.write(f"Match Score: {score}%\n\n")
    buffer.write("✅ Matched Keywords:\n")
    buffer.write(", ".join(sorted(matched)) + "\n\n")
    buffer.write("❌ Missing Keywords:\n")
    buffer.write(", ".join(sorted(missing)) + "\n\n")
    buffer.write("📁 Section-wise Keyword Matches:\n")
    for section, keywords in section_keywords.items():
        buffer.write(f"- {section}: {len(keywords)} matches\n")
        buffer.write("  → " + ", ".join(keywords) + "\n")
    return buffer.getvalue()

def rule_based_suggestions(resume_text, job_keywords):
    suggestions = []
    if len(resume_text) < 500:
        suggestions.append("Resume is too short. Consider elaborating your responsibilities and achievements.")
    if "skill" not in resume_text.lower():
        suggestions.append("Add a dedicated 'Skills' section with relevant tools and technologies.")
    if not job_keywords & set(resume_text.lower().split()):
        suggestions.append("Your resume lacks key terms from the job description.")
    if "project" not in resume_text.lower():
        suggestions.append("Add a 'Projects' section to showcase practical experience.")
    return suggestions or ["Your resume looks well-structured!"]

# Main Logic
if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    clean_resume = clean_text(resume_text)
    match_percent, common, missing, job_keywords = keyword_match(resume_text, job_desc)
    sections = extract_sections(resume_text)

    matched_counts = []
    section_keywords = {}
    for sec, content in sections.items():
        if content.strip():
            words = set(content.lower().split())
            found = words & job_keywords
            matched_counts.append(len(found))
            section_keywords[sec] = list(found)

    # Tabs for UI
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📁 Sections", "📥 Export & Suggestions"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Match Score")
            st.markdown(f"<h2 style='color: green'>{match_percent}%</h2>", unsafe_allow_html=True)
            st.subheader("❌ Missing Keywords")
            if missing:
                st.warning(", ".join(list(missing)[:15]) + ("..." if len(missing) > 15 else ""))
            else:
                st.success("Your resume covers all required keywords!")

        with col2:
            st.subheader("☁️ Resume Word Cloud")
            wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(clean_resume)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    with tab2:
        st.subheader("📊 Keyword Match by Section")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars = ax2.barh(list(section_keywords.keys()), matched_counts, color='#6EC1E4')
        ax2.set_xlabel('Keyword Matches', fontsize=12)
        ax2.set_title('Section-Wise Matches', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax2.bar_label(bars, fmt='%.0f', padding=5)
        st.pyplot(fig2)

        for sec, keywords in section_keywords.items():
            st.markdown(f"**{sec}** – {len(keywords)} keywords matched")
            st.caption(", ".join(keywords) if keywords else "No keywords found.")

    with tab3:
        st.subheader("📥 Downloadable Report")
        report_text = generate_txt_report(match_percent, missing, common, section_keywords)
        b64 = base64.b64encode(report_text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="resume_report.txt">📄 Download Report as TXT</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.subheader("🎯 ATS Suggestions")
        for suggestion in rule_based_suggestions(resume_text, job_keywords):
            st.info("💡 " + suggestion)

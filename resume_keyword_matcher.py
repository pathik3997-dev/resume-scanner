import streamlit as st
import PyPDF2
import re
import string
import openai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="AI Resume Scanner", layout="centered")
st.title("📄 AI Resume Keyword Scanner (Phase 1)")

# --- Load OpenAI API key ---
client = OpenAI(api_key=st.secrets["openai_key"])

# --- PDF Upload & JD Input ---
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📝 Paste the Job Description below:", height=180)

# --- PDF Text Extractor ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# --- GPT Bullet Point Suggestions ---
def gpt_suggestions(missing_keywords, jd):
    prompt = f"""You're a resume expert. Suggest up to 5 bullet points to help a resume include these keywords: {', '.join(missing_keywords)}. The role is based on this job description:\n{jd}"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message.content

# --- Word Cloud Visualizer ---
def show_wordcloud(text):
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- SCAN ---
if st.button("🚀 Scan Resume"):
    if not uploaded_file or not job_desc:
        st.warning("Please upload a resume and paste a job description.")
    else:
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_desc)

        # Cosine Similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        st.success(f"✅ Resume Match Score: **{round(similarity * 100, 2)}%**")

        # Keywords
        resume_words = set(resume_clean.split())
        jd_words = set(jd_clean.split())
        matched = jd_words.intersection(resume_words)
        missing = jd_words.difference(resume_words)

        st.subheader("✅ Matched Keywords")
        st.write(", ".join(list(matched)[:50]))

        st.subheader("❌ Missing Keywords")
        st.write(", ".join(list(missing)[:30]) or "None")

        # GPT Suggestions
        if missing:
            with st.spinner("💡 AI is analyzing your resume..."):
                suggestions = gpt_suggestions(list(missing)[:10], job_desc)
                st.markdown("### 💡 GPT Resume Suggestions")
                st.markdown(suggestions)

        # Word Cloud
        st.markdown("### 🔥 Resume Word Cloud")
        show_wordcloud(resume_clean)

        # Section-wise breakdown
        st.markdown("### 🧩 Section-Wise Breakdown")
        sections = {
            "Skills": re.findall(r"(skills|technologies)[\s\S]{0,300}", resume_text, re.I),
            "Experience": re.findall(r"(experience)[\s\S]{0,1000}", resume_text, re.I),
            "Education": re.findall(r"(education)[\s\S]{0,500}", resume_text, re.I)
        }
        for sec, content in sections.items():
            if content:
                st.markdown(f"**📌 {sec} Section:**")
                st.code(content[0])
            else:
                st.warning(f"❌ No '{sec}' section found!")

        # Resume Preview
        st.markdown("### 📄 Full Resume Text (Extracted)")
        st.text_area("Resume Preview", resume_text, height=250)

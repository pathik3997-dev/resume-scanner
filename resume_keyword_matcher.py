import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI, RateLimitError
import os

# Load OpenAI Key from Streamlit secrets
openai_api_key = st.secrets["openai_key"]
client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="AI Resume Keyword Scanner", layout="wide")

st.title("📄 AI Resume Keyword Scanner & ATS Optimizer")

# 1. Upload Resume
uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type=["pdf"])

# 2. Input Job Description
job_desc = st.text_area("Paste the Job Description Here", height=200)

# Function: Extract text from uploaded resume
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function: Clean and normalize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Function: Extract keywords
def extract_keywords(text, top_n=30):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Function: GPT suggestions
def gpt_suggestions(missing_keywords, jd):
    prompt = f"""You're a resume expert. Suggest up to 5 improvements to add the following keywords to a resume:\n{', '.join(missing_keywords)}\n\nThe job description is:\n{jd}"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except RateLimitError:
        return "⚠️ OpenAI rate limit reached. Please wait and try again later."

# Main: When user submits
if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_desc)

    resume_keywords = extract_keywords(clean_resume)
    jd_keywords = extract_keywords(clean_jd)

    matched = set(resume_keywords) & set(jd_keywords)
    missing = set(jd_keywords) - set(resume_keywords)

    match_percent = round(len(matched) / len(jd_keywords) * 100)

    st.subheader("🔍 Keyword Match Summary")
    st.write(f"✅ **Match:** {len(matched)} / {len(jd_keywords)}")
    st.progress(match_percent)
    st.write(f"🧠 Match Percentage: **{match_percent}%**")

    st.subheader("✅ Matched Keywords")
    st.write(", ".join(matched) if matched else "No keywords matched.")

    st.subheader("❌ Missing Keywords")
    st.write(", ".join(missing) if missing else "No missing keywords!")

    st.subheader("💡 GPT Suggestions to Improve Resume")
    with st.spinner("Generating suggestions..."):
        suggestions = gpt_suggestions(list(missing)[:10], job_desc)
    st.write(suggestions)

    st.subheader("📊 Resume Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_resume)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    st.subheader("📎 Resume Text (Extracted)")
    with st.expander("Click to view extracted resume text"):
        st.write(resume_text)

# resume_keyword_matcher.py

import streamlit as st
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Keyword Scanner", layout="centered")

st.title("🔍 AI Resume Keyword Scanner")
st.write("Upload your resume and paste the job description. We'll tell you how well they match!")

# Upload resume file or paste text
resume_text = st.text_area("📄 Paste your Resume Text here", height=200)
job_desc_text = st.text_area("📝 Paste the Job Description here", height=200)

if st.button("Scan Resume"):
    if not resume_text or not job_desc_text:
        st.warning("Please paste both resume and job description.")
    else:
        def clean_text(text):
            text = text.lower()
            text = re.sub(f"[{string.punctuation}]", "", text)
            return text

        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_desc_text)

        # TF-IDF for matching
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        st.success(f"✅ Resume Match Score: **{round(similarity * 100, 2)}%**")

        # Extract keywords from JD
        jd_keywords = set(jd_clean.split())
        resume_keywords = set(resume_clean.split())

        matched = jd_keywords.intersection(resume_keywords)
        missing = jd_keywords.difference(resume_keywords)

        st.write("### ✅ Matched Keywords:")
        st.write(", ".join(matched) if matched else "None")

        st.write("### ❌ Missing Keywords:")
        st.write(", ".join(missing) if missing else "None")

        st.info("Tip: Add missing relevant keywords to improve your chances with ATS systems.")

        # ATS-Friendly Formatting Tips
        st.write("### 📌 ATS Formatting Suggestions")
        st.markdown("""
        - ✅ Use standard section headers like `Experience`, `Skills`, `Education`
        - ✅ Save your resume as a PDF (unless JD says otherwise)
        - ❌ Avoid using tables or columns (ATS may read it incorrectly)
        - ❌ No fancy fonts, icons, or images
        - ✅ Use simple bullet points and plain text
        """)

        # Resume Fix Suggestions (based on missing keywords)
        if missing:
            suggestions = [
                f"Consider adding keywords like **{word}** to highlight relevant experience."
                for word in list(missing)[:5]
            ]
            st.write("### 💡 Suggested Improvements for ATS:")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")

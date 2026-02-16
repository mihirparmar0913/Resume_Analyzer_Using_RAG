import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import load_and_chunk, create_vectorstore
from skill_extractor import extract_skills
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")  
st.title("Resume–JD Skill Gap Analyzer (RAG)")

resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
jd_text = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if not resume_file or not jd_text:
        st.error("Upload resume and enter JD")
    else:
        with open("resume.pdf", "wb") as f:
            f.write(resume_file.read())

        # Load and chunk resume
        chunks = load_and_chunk("resume.pdf")
        # Get full resume text
        full_resume_text = " ".join([doc.page_content for doc in chunks])
        # Extract skills
        resume_skills = extract_skills(full_resume_text, api_key)
        jd_skills = extract_skills(jd_text, api_key)

        matched = set(resume_skills).intersection(set(jd_skills))
        missing = set(jd_skills) - set(resume_skills)

        if len(jd_skills) > 0:
            match_score = int((len(matched) / len(jd_skills)) * 100)
        else:
            match_score = 0

        # Create vector DB
        vectorstore = create_vectorstore(chunks)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Retrieve relevant content
        relevant_docs = retriever.get_relevant_documents(jd_text)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Prompt
        prompt = f"""
    You are a career assistant.

    Based on the missing skills below:
    {list(missing)}

    And the job description:
    {jd_text}

    Provide:
    1. Suggestions to improve the resume
    2. Learning path for missing skills
"""

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key
        )

        response = llm.invoke(prompt)
        st.subheader("Match Score")
        st.write(f"{match_score}%")

        st.subheader("Matched Skills")
        st.write(list(matched))

        st.subheader("Missing Skills")
        st.write(list(missing))

        st.subheader("Analysis Result")
        st.write(response.content)

        # Simple metric
        st.write(f"Documents retrieved: {len(relevant_docs)}")

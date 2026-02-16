from langchain_google_genai import ChatGoogleGenerativeAI


def extract_skills(text, api_key):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

    prompt = f"""
Extract only the technical skills from the text below.
Return them as a comma-separated list.

Text:
{text}
"""

    response = llm.invoke(prompt)

    skills = response.content.split(",")

    # Clean skills
    return [s.strip().lower() for s in skills if s.strip()]

import google.generativeai as genai
import os

def configure_llm(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def get_llm_response(llm_model, query, document=None):
    if document:
        prompt = f"Truy vấn: {query}\nDựa vào các điều luật sau: {document}."
    else:
        prompt = f"Truy vấn: {query}\nTrả lời như một chuyên gia pháp luật."

    return llm_model.generate_content(prompt).text

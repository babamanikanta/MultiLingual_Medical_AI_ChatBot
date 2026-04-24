import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Try gemini-2.5-flash
    try:
        print("🔄 Testing gemini-2.5-flash...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("I have fever and cough. Explain briefly what might be wrong with me.")
        print(f"\n✅ Success!\n\n{response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("No API key")
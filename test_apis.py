import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Testing API Keys...")
print(f"OpenAI Key Length: {len(os.getenv('OPENAI_API_KEY', ''))}")
print(f"Google Key Length: {len(os.getenv('GOOGLE_API_KEY', ''))}")

# Test OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ OpenAI: Working")
except Exception as e:
    print(f"❌ OpenAI Error: {e}")

# Test Google Gemini
import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
try:
    model = genai.GenerativeModel('gemini-pro')
    print("✅ Google Gemini: Working")
except Exception as e:
    print(f"❌ Google Gemini Error: {e}")

import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-pro")
response = model.generate_content("Test: Generate a simple sentence.")
print(response.text)
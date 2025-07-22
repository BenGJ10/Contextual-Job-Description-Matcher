import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import re
import json
import google.generativeai as genai
from typing import List, Dict, Optional
from src.backend.utils.logger import logging
import dotenv

# Load environment variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key = GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Load skills configuration
with open("config/skills.json", "r") as f:
    SKILLS_CONFIG = json.load(f)

class SkillExtractor:
    def __init__(self):
        self.skills_config = SKILLS_CONFIG

    def extract_skills(self, text: str) -> List[Dict]:
        """
        Extract technical and soft skills from text using Gemini.
        """
        
        try:
            prompt = f"""
            Extract all technical and soft skills from the provided text, mapping synonyms to canonical names in the following skills list:
            {json.dumps(self.skills_config, indent=2)}
            Output *only* a valid JSON array of objects with 'name' and 'category' keys.
            Example: [
                {{"name": "Python", "category": "programming_languages"}},
                {{"name": "Critical Thinking", "category": "soft_skills"}}
            ]
            Use canonical names for synonyms (e.g., 'Python3' or 'Py' → 'Python').
            If no skills are found, return an empty array [].
            Do not include any text outside the JSON array.
            Text: {text[:2000]}
            """
            
            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)
            logging.debug(f"Raw Gemini response: {response.text}")
            cleaned_response = response.text.strip("```json\n```").strip()
            if not cleaned_response:
                logging.warning("Empty response from Gemini")
                return []
            
            try:
                skills = json.loads(cleaned_response)
                if not isinstance(skills, list):
                    logging.warning(f"Gemini response is not a list: {cleaned_response}")
                    return []
                valid_skills = [
                    skill for skill in skills
                    if isinstance(skill, dict) and "name" in skill and "category" in skill
                    and skill["name"] in sum(self.skills_config.values(), [])
                ]
                logging.debug(f"Extracted skills: {valid_skills}")
                return valid_skills
            
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemini response as JSON: {str(e)}")
                
                # Fallback: Extract skills from malformed response
                skills = []
                pattern = r'"name":\s*"([^"]+)",\s*"category":\s*"([^"]+)"'
                matches = re.findall(pattern, cleaned_response)
                
                for name, category in matches:
                    if name in sum(self.skills_config.values(), []):
                        skills.append({"name": name, "category": category})
                logging.debug(f"Fallback extracted skills: {skills}")
                return skills
            
        except Exception as e:
            logging.error(f"Error extracting skills: {str(e)}")
            return []
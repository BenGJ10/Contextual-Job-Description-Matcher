import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import re
import json
import spacy
import textacy
from typing import Dict, List, Optional, Union
from src.backend.utils.logger import logging

# Loading a pretrained English language model
nlp = spacy.load("en_core_web_sm") 

def load_skills_dict() -> Dict[str, List[str]]:
    """
    Loads technical skills from config/skills.json.
    """
    skills_file = "config/skills.json"
    if not os.path.exists(skills_file):
        logging.error(f"Skills file not found: {skills_file}")
        return {}
    try:
        with open(skills_file, "r") as file:
            skills_dict = json.load(file)
            logging.info("Loaded skills dictionary")
        return skills_dict
    except Exception as e:
        logging.error(f"Error loading skills.json: {str(e)}")
        return {}
    
def extract_skills(doc_data: Dict[str, Union[int, str]]) -> Optional[Dict[str, any]]:
    """
    Extract technical skills from document text and detect sections.
    """
    try:
        doc_id = doc_data["doc_id"]
        doc_type = doc_data["doc_type"]
        text = doc_data["text"]
        logging.info(f"Extracting skills from {doc_type} (ID: {doc_id})")

        # Loading skill dict
        skills_dict = load_skills_dict()
        if not skills_dict:   
            return None
        # Processing text with SpaCy
        doc = nlp(text)

        # Extracting key terms with textacy and converts raw text into a SpaCy document
        doc_textacy = textacy.make_spacy_doc(text, lang = "en_core_web_sm")
        # Textrank lemmatizes words and ranks important key phrases then return top20 key terms
        terms = textacy.extract.keyterms.textrank(doc_textacy, normalize = "lemma", topn = 20)

        # Initialize skills list
        extracted_skills = []

        # Matching terms against skills dict
        for term, _ in terms:
            for category, skills in skills_dict.items():
                for skill in skills:
                    if skill.lower() in term.lower():
                        extracted_skills.append({
                            "name": skill,
                            "category": category,
                            "match": None  # Placeholder for Multi-Job Matching
                        })

        # Additional NER-based extraction
        for ent in doc.ents: # doc_ents contains all named entities SpaCy has recognized      
            if ent.label_ in ["ORG", "PRODUCT"] and any(
                ent.text.lower() in skill.lower() for skill in sum(skills_dict.values(), [])
            ):
                for category, skills in skills_dict.items():
                    for skill in skills:
                        if skill.lower() in ent.text.lower():
                            extracted_skills.append({
                                "name": skill,
                                "category": category,
                                "match": None
                            })

        # Removing duplicates
        unique_skills = []
        seen = set()
        for skill in extracted_skills:
            if skill["name"] not in seen:
                unique_skills.append(skill)
                seen.add(skill["name"])

        # Detecting sections for Resume Quality Score (clarity) using regex
        sections = []
        section_patterns = [
            r"\bskills\b",          # matches the word "skills" only if it stands alone
            r"\bexperience\b",
            r"\beducation\b",
            r"\bcertifications\b"
        ]
        for pattern in section_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sections.append(pattern.replace(r"\b", ""))
        
        logging.info(f"Extracted {len(unique_skills)} skills and {len(sections)} sections from {doc_type} (ID: {doc_id})")

        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "skills": unique_skills,
            "sections": sections,
            "keyword_density": len(unique_skills) / doc_data["word_count"] if doc_data["word_count"] > 0 else 0
        }
    except Exception as e:
        logging.error(f"Error extracting skills for {doc_id}: {str(e)}")
        return None
    
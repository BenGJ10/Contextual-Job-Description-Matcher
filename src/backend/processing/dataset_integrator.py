import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.backend.utils.logger import logging
from src.backend.processing.document_processor import extract_text
from src.backend.processing.skill_extractor import extract_skills
from src.backend.utils.data_formatter import format_data


def compute_ats_score(skill_data: Dict[str, any], word_count: int) -> float:
    """
    Compute ATS compatibility score based on keyword density, clarity, and formatting.
    Returns:
        float: ATS score (0-100).
    """
    try:
        keyword_density = skill_data.get("keyword_density", 0)
        sections = skill_data.get("sections", [])
        expected_sections = ["skills", "experience", "education", "certifications", "publications", "projects"]
        section_score = len([s for s in sections if s.lower() in expected_sections]) / len(expected_sections)
        formatting_score = 1.0 if 200 <= word_count <= 1000 else 0.5  # Penalize if outside optimal range
        ats_score = (keyword_density * 0.5 + section_score * 0.3 + formatting_score * 0.2) * 100
        ats_score = round(ats_score, 2)  # Optional rounding

        logging.debug(f"ATS score components: density={keyword_density}, sections={section_score}, formatting={formatting_score}")
        return ats_score
    except Exception as e:
        logging.error(f"Error computing ATS score: {str(e)}")
        return 0.0
    
def match_resume_to_jobs(resume_skills: List[Dict], job_skills: List[Dict[str, List[Dict]]]) -> List[Dict]:
    """
    Match resume skills to multiple job descriptions using cosine similarity.
    """
    try:
        # Extract skill names from resume and job descriptions
        resume_skill_names = []
        for skill in resume_skills:
            resume_skill_names.append(skill["name"])
         
        job_skill_texts = []
        job_ids = []
        for job in job_skills:
            job_skill_names = []
            for skill in job["skills"]:
                job_skill_names.append(skill["name"])
            job_skill_texts.append(" ".join(job_skill_names)) # For each job, join skill names into one string
            job_ids.append(job["doc_id"])
        
        # Vectorize Skills
        vectorizer = CountVectorizer() 
        skill_vectors = vectorizer.fit_transform([ " ".join(resume_skill_names)] + job_skill_texts) # First vector is the resume skills
        resume_vector = skill_vectors[0]
        job_vectors = skill_vectors[1:]
        
        # Compute cosine similarity
        similarities = cosine_similarity(resume_vector, job_vectors)[0]
        matches = [
            {"job_id": job_ids[i], "match_score": float(similarities[i] * 100)}
            for i in range(len(job_ids))
        ]
        
        # Sort by match score (descending)
        matches = sorted(matches, key = lambda x: x["match_score"], reverse = True)
        logging.info(f"Computed {len(matches)} job matches for resume")
        return matches[:3]  # Limit to top 3 matches
    
    except Exception as e:
        logging.error(f"Error matching resume to jobs: {str(e)}")
        return []

def process_dataset(resume_dir: str = "data/resumes", job_dir: str = "data/jobs") -> List[Dict]:
    """
    Process resumes and job descriptions, integrating previous steps.
    """
    results = []
    job_data_list = []

    try:
        # Process job descriptions
        os.makedirs(job_dir, exist_ok=True)
        for file_name in os.listdir(job_dir):
            file_path = os.path.join(job_dir, file_name)
            if not (file_path.endswith(".docx") or file_path.endswith(".pdf")):
                continue
            doc_data = extract_text(file_path, "job")
            if not doc_data:
                continue
            skill_data = extract_skills(doc_data)
            if not skill_data:
                continue
            formatted_data = format_data(doc_data, skill_data)
            if formatted_data:
                job_data_list.append(formatted_data)
                logging.info(f"Processed job description: {file_path}")

        # Process resumes
        os.makedirs(resume_dir, exist_ok=True)
        for file_name in os.listdir(resume_dir):
            file_path = os.path.join(resume_dir, file_name)
            if not (file_path.endswith(".docx") or file_path.endswith(".pdf")):
                continue
            doc_data = extract_text(file_path, "resume")
            if not doc_data:
                logging.warning(f"Skipping resume due to extraction failure: {file_path}")
                continue
            skill_data = extract_skills(doc_data)
            if not skill_data:
                logging.warning(f"Skipping resume due to skill extraction failure: {file_path}")
                continue
            formatted_data = format_data(doc_data, skill_data)
            if not formatted_data:
                logging.warning(f"Skipping resume due to formatting failure: {file_path}")
                continue

            # Compute ATS score
            ats_score = compute_ats_score(skill_data, doc_data["word_count"])
            formatted_data["metrics"]["ats_score"] = ats_score
                
            # Match to jobs (if jobs exist)
            if job_data_list:
                matches = match_resume_to_jobs(formatted_data["skills"], job_data_list)
                formatted_data["job_matches"] = matches
        
            results.append(formatted_data)
            logging.info(f"Processed resume: {file_path} with ATS score: {ats_score}")
    
        return results
    
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        return []
        
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import glob
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import CountVectorizer
from src.backend.utils.logger import logging
from src.backend.processing.document_processor import DocumentProcessor
from src.backend.processing.skill_extractor import SkillExtractor
from src.backend.utils.data_formatter import format_data
import google.generativeai as genai
import dotenv

# Load environment variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY not found in .env")

class DatasetIntegrator:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.skill_extractor = SkillExtractor()
        with open("config/skills.json", "r") as f:
            self.skills_config = json.load(f)

    def compute_relevance_score(self, resume_skills: List[Dict], jd_skills: List[Dict], resume_text: str, jd_text: str) -> float:
        """Compute relevance score based on skill overlap and Gemini text similarity."""
        try:
            resume_skill_names = set(skill["name"] for skill in resume_skills)
            jd_skill_names = set(skill["name"] for skill in jd_skills)
            overlap = len(resume_skill_names & jd_skill_names) / len(jd_skill_names) if jd_skill_names else 0
            overlap_score = overlap * 50

            prompt = f"""
            Compare the following resume and job description for semantic similarity.
            Return a score (0-100) based on how well the resume matches the job requirements.
            Resume: {resume_text}
            Job Description: {jd_text}
            """
            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)
            similarity_score = float(response.text.strip()) if response.text.isdigit() else 0
            return (overlap_score + similarity_score) / 2
        
        except Exception as e:
            logging.error(f"Error computing relevance score: {str(e)}")
            return 0.0

    def compute_completeness_score(self, resume_skills: List[Dict]) -> float:
        """Compute completeness score based on coverage of skills.json domains."""
        try:
            resume_skill_names = set(skill["name"] for skill in resume_skills)
            all_skills = set()
            for category, skills in self.skills_config.items():
                if category == "soft_skills":
                    continue
                all_skills.update(skills)
            coverage = len(resume_skill_names & all_skills) / len(all_skills) if all_skills else 0
            return coverage * 100
        
        except Exception as e:
            logging.error(f"Error computing completeness score: {str(e)}")
            return 0.0

    def process_dataset(self, resume_dir: str = "data/resumes", job_dir: str = "data/jobs") -> List[Dict]:
        """Process resumes and JDs, extracting skills and computing metrics."""
        try:
            results = []
            resume_files = glob.glob(os.path.join(resume_dir, "*")) # Assuming resumes are in a flat structure
            job_files = glob.glob(os.path.join(job_dir, "*")) # Assuming jobs are in a flat structure

            # Process job descriptions
            job_data_list = []
            for job_file in job_files:
                doc_data = self.doc_processor.extract_text(job_file, doc_type = "job")
                if not doc_data:
                    logging.warning(f"No text extracted from {job_file}")
                    continue

                skills = self.skill_extractor.extract_skills(doc_data["text"])
                job_data = format_data(doc_data, skills)
                
                if not job_data:
                    logging.warning(f"Failed to format data for {job_file}")
                    continue
                job_data_list.append(job_data)
                results.append(job_data)

            
           # Process resumes
            for resume_file in resume_files:
                doc_data = self.doc_processor.extract_text(resume_file, doc_type = "resume")
                if not doc_data:
                    logging.warning(f"No text extracted from {resume_file}")
                    continue
                skills = self.skill_extractor.extract_skills(doc_data["text"])
                resume_data = format_data(doc_data, skills)
                if not resume_data:
                    logging.warning(f"Failed to format data for {resume_file}")
                    continue
                
                # Compute relevance score against the most relevant JD
                max_relevance_score = 0.0
                if job_data_list:
                    resume_skill_names = set(skill["name"] for skill in skills)
                    for job_data in job_data_list:
                        jd_skill_names = set(skill["name"] for skill in job_data["skills"])
                        overlap = len(resume_skill_names & jd_skill_names) / len(jd_skill_names) if jd_skill_names else 0
                        score = self.compute_relevance_score(skills, job_data["skills"], doc_data["text"], job_data["text"])
                        max_relevance_score = max(max_relevance_score, score * 0.5 + overlap * 50)
                
                resume_data["metrics"] = {
                    "relevance_score": max_relevance_score,
                    "completeness_score": self.compute_completeness_score(skills)
                }
                
                resume_data["job_matches_rag"] = []
                results.append(resume_data)

            logging.info(f"Processed {len(results)} documents")
            return results
        
        except Exception as e:
            logging.error(f"Error processing dataset: {str(e)}")
            return []
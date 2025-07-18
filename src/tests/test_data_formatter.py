import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.backend.processing.document_processor import extract_text
from src.backend.processing.skill_extractor import extract_skills
from src.backend.utils.data_formatter import format_data

if __name__ == "__main__":
    files = [
        ("data/resumes/sample_resume.pdf", "resume"),
        ("data/resumes/sample_resume.docx", "resume")
    ]
    
    for file_path, doc_type in files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Extract text
        doc_data = extract_text(file_path, doc_type)
        if not doc_data:
            print(f"Failed to extract text from: {file_path}")
            continue
        
        # Extract skills
        skill_data = extract_skills(doc_data)
        if not skill_data:
            print(f"Failed to extract skills from: {file_path}")
            continue
        
        # Format data
        result = format_data(doc_data, skill_data)
        if result:
            print(f"\nFormatted data for {file_path} (ID: {result['doc_id']}):")
            print(result)
        else:
            print(f"\nFailed to format data for: {file_path}")
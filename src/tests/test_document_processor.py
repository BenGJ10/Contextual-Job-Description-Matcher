import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.backend.processing.document_processor import extract_text

if __name__ == "__main__":
    files = [
        ("data/resumes/sample_resume.pdf", "resume"),
        ("data/resumes/sample_resume.docx", "resume")
    ]

    for file_path, doc_type in files:
        result = extract_text(file_path, doc_type)
        if result:
            print(f"\nJSON for {file_path}:")
            print(result)
        else:
            print(f"\nFailed to process: {file_path}")
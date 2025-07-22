import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
from src.backend.rag.rag_pipeline import RAGPipeline
from src.backend.utils.logger import logging

if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Process dataset with one resume and one JD
    results = rag.process_rag(resume_dir="data/resumes", job_dir="data/jobs")
    
    for result in results:
        if result["doc_type"] == "resume":
            logging.info(f"Processed RAG data for resume (ID: {result['doc_id']}):")
            print(json.dumps(result, indent=2))
        else:
            logging.info(f"Processed job (ID: {result['doc_id']})")
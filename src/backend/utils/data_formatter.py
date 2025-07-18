import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import boto3
import textstat
from typing import Dict, Optional
from botocore.exceptions import ClientError
from src.backend.utils.logger import logging

def format_data(doc_data: Dict[str, any], skill_data: Dict[str, any]) -> Optional[Dict[str, any]]:
    """
    Takes doc_data anf skill_data JSON objects and merges them into single structured JSON format and compute readability and clarity scores.
    """
    try:
        # Makes sure both inputs are present and refer to the same document
        if not doc_data or not skill_data or doc_data["doc_id"] != skill_data["doc_id"]:
            logging.error("Invalid or mismatched input data")
            return None
        
        doc_id = doc_data["doc_id"]
        doc_type = doc_data["doc_type"]
        text = doc_data["text"]
        word_count = doc_data["word_count"]
        skills = skill_data.get("skills", [])
        sections = skill_data.get("sections", [])

        logging.info(f"Formatting data for {doc_type} (ID: {doc_id})")

        # Computing the readability score
        readability = textstat.flesch_reading_ease(text)
        

        # Compute clarity score (based on sections and skills detected)
        expected_sections = ["skills", "experience", "education", "certifications", "publications", "projects"]
        clarity_score = len([s for s in sections if s.lower() in expected_sections]) / len(expected_sections)

         # Unified JSON format
        formatted_data = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "text": text,
            "word_count": word_count,
            "skills": skills,
            "metrics": {
                "keyword_density": skill_data.get("keyword_density", 0),
                "clarity_score": clarity_score,
                "readability_score": readability
            }
        }
    
        # Boto3 is the official AWS SDK for Python. 
        # Uses your local AWS credentials, finds the bucket, uploads the local file to the key inside that bucket.
        os.makedirs("data/processed", exist_ok=True) # Ensure the directory exists
        local_path = f"data/processed/{doc_id}.json" # Save to local file

        try:
            with open(local_path, "w") as f:
                json.dump(formatted_data, f, indent = 2) # Save formatted data to local file
            logging.info(f"Saved formatted data to {local_path}")

            # Simulate S3 upload with boto3 
            s3_client = boto3.client("s3")
            try:
                with open (local_path, "rb") as f:
                    s3_client.upload_fileobj(f, "job-matcher", f"formatted_docs/{doc_id}.json")
                    logging.info(f"Simulated S3 upload for {doc_id}.json")
            except Exception as e:
                logging.error(f"Failed to simulate S3 upload: {str(e)}")
        
        except Exception as e:
            logging.error(f"Error saving formatted data: {str(e)}")
        

        logging.info(f"Formatted data for {doc_type} with {len(skills)} skills")
        return formatted_data
    
    except Exception as e:
        logging.error(f"Error formatting data for {doc_id}: {str(e)}")
        return None
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import boto3
from typing import Dict, List, Optional
from src.backend.utils.logger import logging



def format_data(doc_data: Dict[str, any], skills: List[Dict]) -> Optional[Dict[str, any]]:
    """
    Merge doc_data and skills into a single structured JSON format.
    
    Args:
        doc_data (Dict): Document data with doc_id, doc_type, text, word_count, file_name, file_size_mb.
        skills (List[Dict]): List of skills with name and category.
    
    Returns:
        Optional[Dict]: Formatted JSON data, or None if error.
    """
    try:
        # Validate inputs
        if not doc_data or not isinstance(skills, list):
            logging.error("Invalid input data or skills")
            return None
        
        doc_id = doc_data["doc_id"]
        doc_type = doc_data["doc_type"]
        text = doc_data["text"]
        word_count = doc_data["word_count"]
        file_name = doc_data.get("file_name", "")
        file_size_mb = doc_data.get("file_size_mb", 0.0)

        logging.info(f"Formatting data for {doc_type} (ID: {doc_id})")

        # Unified JSON format
        formatted_data = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "text": text,
            "word_count": word_count,
            "skills": skills,
            "file_name": file_name,
            "file_size_mb": file_size_mb
        }

        # Save to local file
        os.makedirs("data/processed", exist_ok=True)
        local_path = f"data/processed/{doc_id}.json"
        try:
            with open(local_path, "w") as f:
                json.dump(formatted_data, f, indent=2)
            logging.info(f"Saved formatted data to {local_path}")

            # Simulate S3 upload
            s3_client = boto3.client("s3")
            try:
                with open(local_path, "rb") as f:
                    s3_client.upload_fileobj(f, "job-matcher", f"formatted_docs/{doc_id}.json")
                logging.info(f"Simulated S3 upload for {doc_id}.json")
            except Exception as e:
                logging.error(f"Failed to simulate S3 upload: {str(e)}")
        
        except Exception as e:
            logging.error(f"Error saving formatted data: {str(e)}")
            return None

        logging.info(f"Formatted data for {doc_type} with {len(skills)} skills")
        return formatted_data
    
    except Exception as e:
        logging.error(f"Error formatting data for {doc_id}: {str(e)}")
        return None
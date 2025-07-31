import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import json
import time
import boto3
from typing import Dict, List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from src.backend.utils.logger import logging
from botocore.exceptions import ClientError


def format_data(doc_data: Dict[str, any], skills: List[Dict]) -> Optional[Dict[str, any]]:
    """
    Merge doc_data and skills into a single structured JSON format.
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
            "file_size_mb": file_size_mb,
            "created_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            "processed_by": "v1.0"
        }

        if doc_type == "job":
            formatted_data["job_title"] = doc_data.get("job_title", "")
            formatted_data["company"] = doc_data.get("company", "")


        # Save to local file
        os.makedirs("data/processed", exist_ok=True)
        local_path = f"data/processed/{doc_id}.json"
        try:
            with open(local_path, "w") as f:
                json.dump(formatted_data, f, indent=2)
            logging.info(f"Saved formatted data to {local_path}")
        except Exception as e:
            logging.error(f"Error saving formatted data locally: {str(e)}")
            return None
          
        # Initialize S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id  = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        bucket_name = os.getenv("S3_BUCKET", "job-matcher-proj")

         # Upload JSON to S3
        for attempt in range(3):
            try:
                with open(local_path, "rb") as f:
                    s3_client.upload_fileobj(f, bucket_name, f"processed/{doc_id}.json")
                logging.info(f"Uploaded JSON to S3: processed/{doc_id}.json")
                break
            except ClientError as e:
                logging.warning(f"S3 JSON upload attempt {attempt+1}/3 failed: {str(e)}")
                if attempt == 2:
                    logging.error(f"Failed to upload JSON to S3: {str(e)}")
                    return None
                time.sleep(1)

        # Cache JSON locally
        os.makedirs("data/cache", exist_ok = True)
        cache_path = f"data/cache/{doc_id}.json"
        try:
            with open(cache_path, "w") as f:
                json.dump(formatted_data, f, indent=2)
            logging.info(f"Cached JSON locally: {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to cache JSON locally: {str(e)}")

        logging.info(f"Formatted data for {doc_type} with {len(skills)} skills")
        return formatted_data
    
    except Exception as e:
        logging.error(f"Error formatting data for {doc_id}: {str(e)}")
        return None
import os
import json
import time
import boto3
from typing import Dict, Optional
from botocore.exceptions import ClientError
from src.backend.utils.logger import logging

def get_formatted_data(doc_id: str) -> Optional[Dict]:
    """
    Retrieve formatted JSON data by doc_id, checking local cache first, then S3.
    """
    try:
        # Check local cache (TTL: 24 hours)
        cache_path = f"data/cache/{doc_id}.json"
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < 86400:
            try:
                with open(cache_path, "r") as f:
                    logging.info(f"Retrieved {doc_id} from local cache")
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to read cache for {doc_id}: {str(e)}")

        # Initialize S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        bucket_name = os.getenv("S3_BUCKET", "job-matcher-proj")

        # Fetch from S3
        for attempt in range(3):
            try:
                s3_client.download_file(bucket_name, f"processed/{doc_id}.json", cache_path)
                with open(cache_path, "r") as f:
                    data = json.load(f)
                logging.info(f"Retrieved {doc_id} from S3 and cached locally")
                return data
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    logging.error(f"JSON not found in S3: processed/{doc_id}.json")
                    return None
                logging.warning(f"S3 download attempt {attempt+1}/3 failed: {str(e)}")
                if attempt == 2:
                    logging.error(f"Failed to fetch {doc_id} from S3: {str(e)}")
                    return None
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error reading downloaded JSON for {doc_id}: {str(e)}")
                return None
    
    except Exception as e:
        logging.error(f"Unexpected error retrieving {doc_id}: {str(e)}")
        return None
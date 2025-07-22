import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) 

import json
from typing import Dict, List, Optional
import google.generativeai as genai
import chromadb
from src.backend.utils.logger import logging
from src.backend.processing.dataset_integrator import process_dataset
import dotenv

# Loading environment variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key = GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Generate embeddings, store in Chroma, and perform similarity search for Multi-Job Matching.
class RAGPipeline:
    def __init__(self, collection_name: str = "job_matcher"):
        """
        Initialize Chroma client and collection.
        """
        try:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name = collection_name,
                embedding_function = None # Use Gemini embeddings directly
            )
            logging.info(f"Initialized Chrome collection: {collection_name}")

        except Exception as e:
            logging.error(f"Error initializing Chroma: {str(e)}")
            raise 

    def generate_embeddings(self, skills: List[Dict]) -> Optional[List[float]]:
        """
        Generate embeddings for skills using Gemini. Embeddings are used for similarity search.
        """
        try:
            skill_text = " ".join([skill["name"] for skill in skills]) # Join skill names into a single string
            if not skill_text:
                logging.warning("No skills provided for embedding")
                return None
            
            result = genai.embed_content(
                model = "models/embedding-001",  # Gemini embedding model
                content = skill_text,
                task_type = "semantic_similarity" # Specify task type for embeddings
            )
            embeddings = result["embedding"] # Extract embeddings from the result
            logging.debug(f"Generated embeddings for skills: {skill_text}")
            return embeddings 
        
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def store_document(self, doc_data: Dict[str, any]) -> bool:
        """
        Store document embeddings in Chroma.
        """
        try:
            doc_id = doc_data["doc_id"]
            skills = doc_data["skills"]
            embeddings = self.generate_embeddings(skills)
            if not embeddings:
                logging.warning(f"No embeddings generated for doc_id: {doc_id}")
                return False
            
            self.collection.add( # Add document to Chroma collection
                documents = [json.dumps(doc_data)],
                embeddings = [embeddings],
                ids = [doc_id],
                metadatas = [{"doc_type": doc_data["doc_type"]}] # Optional metadata
            )
            logging.info(f"Stored embeddings for doc_id: {doc_id}")
            return True
        
        except Exception as e:
            logging.error(f"Error storing document {doc_id}: {str(e)}")
            return False
        
    
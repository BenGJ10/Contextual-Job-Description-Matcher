import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) 

import json
from typing import Dict, List, Optional
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from src.backend.utils.logger import logging
from src.backend.processing.dataset_integrator import DatasetIntegrator
import dotenv

# Loading environment variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key = GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Generate embeddings using Gemini, store in ChromaDB, and perform similarity search for Multi-Job Matching.

class GeminiEmbeddingFunction:
    """
    Embedding function using Google Gemini for text embeddings.
    """
    def __call__(self, texts):
        try:
            # Generate embeddings using Gemini
            result = genai.embed_content(
                model = "models/embedding-001", 
                content = texts,
                task_type = "semantic_similarity" # Assuming semantic similarity task
            )
            # Convert result to list if it's not already
            embeddings = result["embedding"] if isinstance(result["embedding"], list) else [result["embedding"]]
            logging.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            return [[] for _ in texts]

    def embed_documents(self, texts):
        return self.__call__(texts)

    def embed_query(self, text):
        # Chroma expects a list, but for a single query, wrap in a list and return the first embedding
        return self.__call__([text])[0]

class RAGPipeline:
    def __init__(self, collection_name: str = "job_matcher"):
        """
        Initialize Chroma with LangChain and Gemini embeddings.
        """
        try:
            self.embedding_function = GeminiEmbeddingFunction()
            self.vector_store = Chroma(
                collection_name = collection_name,
                embedding_function = self.embedding_function,
                persist_directory = "./chroma_db" # Directory to persist embeddings
            )
            logging.info(f"Initialized Chroma collection: {collection_name}")
        
        except Exception as e:
            logging.error(f"Error initializing Chroma: {str(e)}")
            raise

    def generate_suggestions(self, resume_text: str, jd_text: str, resume_skills: List[Dict], jd_skills: List[Dict]) -> Dict:
        """
        Generate match score, missing skills, and suggestions using Gemini.
        """
        try:
            resume_skill_names = set(skill["name"] for skill in resume_skills)
            jd_skill_names = set(skill["name"] for skill in jd_skills)
            missing_skills = list(jd_skill_names - resume_skill_names)
            
            prompt_template = PromptTemplate.from_template("""
            Compare the following resume and job description for a data science role. Output JSON with:
            - match_score (0-100): Based on skill overlap and text similarity, prioritizing data science skills (e.g., Python, SQL, TensorFlow).
            - missing_skills: Skills in JD but not in resume.
            - suggestions: Tailored advice to improve resume for JD (e.g., add missing skills, highlight relevant projects like LSTM or machine learning work).
            Resume: {resume_text}
            Job Description: {jd_text}
            Resume Skills: {resume_skills}
            JD Skills: {jd_skills}
            """)
            prompt = prompt_template.format(
                resume_text = resume_text[:1000],
                jd_text = jd_text[:1000],
                resume_skills = json.dumps(resume_skills),
                jd_skills = json.dumps(jd_skills)
            )
            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip("```json\n```")) if response.text else {}
            result["missing_skills"] = missing_skills
            logging.debug(f"Generated suggestions: {result}")
            return result
        
        except Exception as e:
            logging.error(f"Error generating suggestions: {str(e)}")
            return {"match_score": 0.0, "missing_skills": missing_skills, "suggestions": "Unable to generate suggestions"}

    def store_document(self, doc_data: Dict[str, any]) -> bool:
        """
        Store JD embeddings in Chroma.
        """
        try:
            doc_id = doc_data["doc_id"]
            skills = doc_data["skills"]
            skill_text = " ".join([skill["name"] for skill in skills])
            if not skill_text:
                logging.warning(f"No skills for doc_id: {doc_id}")
                return False
            self.vector_store.add_texts(
                texts=[skill_text],
                metadatas=[{"doc_id": doc_id, "doc_type": doc_data["doc_type"], "file_name": doc_data.get("file_name", "")}],
                ids=[doc_id]
            )
            logging.info(f"Stored embeddings for doc_id: {doc_id}")
            return True
        
        except Exception as e:
            logging.error(f"Error storing document {doc_id}: {str(e)}")
            return False

    def match_resume(self, resume_data: Dict[str, any], job_data_list: List[Dict]) -> List[Dict]:
        """
        Match resume to JDs using Chroma similarity search and Gemini suggestions.
        """
        try:
            resume_id = resume_data["doc_id"]
            resume_skill_text = " ".join([skill["name"] for skill in resume_data["skills"]])
            if not resume_skill_text:
                logging.warning(f"No skills for resume {resume_id}")
                return []
            
            results = self.vector_store.similarity_search_with_score(resume_skill_text, k=5)
            logging.debug(f"Chroma query results for resume {resume_id}: {[(doc.metadata, score) for doc, score in results]}")
            
            matches = []
            for doc, score in results:
                job_id = doc.metadata["doc_id"]
                job_data = next((jd for jd in job_data_list if jd["doc_id"] == job_id), None)
                if not job_data:
                    logging.warning(f"Job data not found for job_id: {job_id}")
                    continue
                suggestions = self.generate_suggestions(
                    resume_data["text"], job_data["text"], resume_data["skills"], job_data["skills"]
                )
                matches.append({
                    "job_id": job_id,
                    "match_score": float((1 - score) * 100),
                    "missing_skills": suggestions.get("missing_skills", []),
                    "suggestions": suggestions.get("suggestions", "")
                })

            logging.info(f"Found {len(matches)} job matches for resume {resume_id}")
            return sorted(matches, key=lambda x: x["match_score"], reverse = True)
        
        except Exception as e:
            logging.error(f"Error matching resume {resume_id}: {str(e)}")
            return []

    def process_rag(self, resume_dir: str = "data/resumes", job_dir: str = "data/jobs") -> List[Dict]:
        """
        Process dataset through RAG pipeline.
        """
        try:
            dataset_integrator = DatasetIntegrator()
            logging.debug(f"Dataset integrator type: {type(dataset_integrator)}")
            dataset_results = dataset_integrator.process_dataset(resume_dir, job_dir)
            if not dataset_results:
                logging.warning("No dataset results to process")
                return []
            
            results = []
            job_data_list = [r for r in dataset_results if r["doc_type"] == "job"]
            resume_data_list = [r for r in dataset_results if r["doc_type"] == "resume"]

            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name = "job_matcher",
                embedding_function = self.embedding_function,
                persist_directory="./chroma_db"
            )
            logging.info("Reset Chroma collection to ensure fresh JD embeddings")

            for job_data in job_data_list:
                success = self.store_document(job_data)
                if not success:
                    logging.warning(f"Failed to store embeddings for job {job_data['doc_id']}")

            for resume_data in resume_data_list:
                matches = self.match_resume(resume_data, job_data_list)
                resume_data["job_matches_rag"] = matches
                # Ensure metrics field is preserved or recomputed if missing
                if "metrics" not in resume_data:
                    logging.warning(f"No metrics found for resume {resume_data['doc_id']}, recomputing...")
                    if job_data_list:
                        resume_data["metrics"] = {
                            "relevance_score": DatasetIntegrator().compute_relevance_score(
                                resume_data["skills"], job_data_list[0]["skills"], resume_data["text"], job_data_list[0]["text"]
                            ),
                            "completeness_score": DatasetIntegrator().compute_completeness_score(resume_data["skills"])
                        }
                    else:
                        resume_data["metrics"] = {"relevance_score": 0.0, "completeness_score": 0.0}
                
                with open(f"data/processed/{resume_data['doc_id']}.json", "w") as f:
                    json.dump(resume_data, f, indent=2)
                logging.info(f"Processed RAG for resume {resume_data['doc_id']} with {len(matches)} matches")

            logging.info("RAG pipeline processing completed successfully")
            return results
        
        except Exception as e:
            logging.error(f"Error processing RAG pipeline: {str(e)}")
            return []
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))) 

import boto3
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, List, Optional
from src.backend.utils.logger import logging
from src.backend.processing.document_processor import DocumentProcessor
from src.backend.processing.skill_extractor import SkillExtractor
from src.backend.utils.data_formatter import format_data
from src.backend.utils.s3_utils import get_formatted_data
from src.backend.rag.rag_pipeline import RAGPipeline

app = FastAPI(title = "Job Matcher API")

# Initialize pipeline components
doc_processor = DocumentProcessor()
skill_extractor = SkillExtractor(skills_config = json.load(open("config/skills.json")))
rag_pipeline = RAGPipeline()

@app.post("/upload/resume")
async def upload_resume(file: UploadFile = File(...)): # File(...) parameter indicates a required file upload
    """
    Upload and process a resume file (.pdf or .docx).
    """
    try:
        # Checks whether the uploaded file is neither PDF nor DOCX.
        if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            raise HTTPException(status_code = 400, detail = "Invalid file format! Only .pdf or .docx files are allowed.")
        
        file_ext = os.path.splitext(file.filename)[1].lower() # Extracts the file extension
        temp_path = f"data/temp/{file.filename}" # where to temporarily store the uploaded file locally
        os.makedirs("data/temp", exist_ok = True)

        with open(temp_path, "wb") as f:
            f.write(await file.read()) # reads the entire uploaded file asynchronously

        if os.path.getsize(temp_path) / (1024 * 1024) > 5: # If the file is larger than 5 MB, it deletes the file and raise Error
            os.remove(temp_path)
            raise HTTPException(status_code = 400, detail = "File size exceeds 5MB!")
        
        doc_data = doc_processor.extract_text(temp_path, doc_type = "resume") # read PDF/DOCX and extract raw text
        if not doc_data: # If text extraction fails
            os.remove(temp_path)
            raise HTTPException(status_code = 400, detail="Failed to extract text from resume")
        
        skills = skill_extractor.extract_skills(doc_data["text"]) # Extract skills
        
        formatted_data = format_data(doc_data, skills, raw_file_path = temp_path) # converting to structured JSON
        os.remove(temp_path) # Deletes the temporary file from data/temp after processing 
        
        if not formatted_data:
            raise HTTPException(status_code=500, detail="Failed to format resume data")
        
        return {
            "doc_id": formatted_data["doc_id"],
            "doc_type": "resume",
            "skills": formatted_data["skills"],
            "status": "Uploaded and processed successfully"
        }
    
    except Exception as e:
        logging.error(f"Error uploading resume: {str(e)}")
        raise HTTPException(status_code = 500, detail = f"Error processing resume: {str(e)}") # 500 -> Internal Server Error
    

@app.post("/upload/job")
async def upload_job(file: UploadFile = File(...)):
    """
    Upload and process a job description file (.pdf or .docx).
    """
    try:
         # Checks whether the uploaded file is neither PDF nor DOCX.
        if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            raise HTTPException(status_code = 400, detail = "Invalid file format! Only .pdf or .docx files are allowed.")
        
        file_ext = os.path.splitext(file.filename)[1].lower() # Extracts the file extension
        temp_path = f"data/temp/{file.filename}" # where to temporarily store the uploaded file locally
        os.makedirs("data/temp", exist_ok = True)

        with open(temp_path, "wb") as f:
            f.write(await file.read()) # reads the entire uploaded file asynchronously

        if os.path.getsize(temp_path) / (1024 * 1024) > 5: # If the file is larger than 5 MB, it deletes the file and raise Error
            os.remove(temp_path)
            raise HTTPException(status_code = 400, detail = "File size exceeds 5MB!")
        
        doc_data = doc_processor.extract_text(temp_path, doc_type = "job") # read PDF/DOCX and extract raw text
        if not doc_data: # If text extraction fails
            os.remove(temp_path)
            raise HTTPException(status_code = 400, detail="Failed to extract text from job description")
        
        skills = skill_extractor.extract_skills(doc_data["text"]) # Extract skills
        
        formatted_data = format_data(doc_data, skills, raw_file_path = temp_path) # converting to structured JSON
        os.remove(temp_path) # Deletes the temporary file from data/temp after processing 

        if not formatted_data:
            raise HTTPException(status_code = 500, detail = "Failed to format job data")
        
        # Add to Chroma
        rag_pipeline.store_document(formatted_data)
        return {
            "doc_id": formatted_data["doc_id"],
            "doc_type": "job",
            "job_title": formatted_data.get("job_title", ""),
            "company": formatted_data.get("company", ""),
            "skills": formatted_data["skills"],
            "status": "Uploaded and processed successfully"
        }
    except Exception as e:
        logging.error(f"Error uploading job description: {str(e)}")
        raise HTTPException(status_code = 500, detail = f"Error processing job description: {str(e)}")



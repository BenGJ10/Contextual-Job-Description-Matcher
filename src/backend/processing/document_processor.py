# Extract text from PDF and .docx files, outputting JSON with metadata

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import re
import uuid # Unique identifier for documents
import PyPDF2 # PDF processor
from docx import Document # Document processor
from typing import Dict, Optional 
from src.backend.utils.logger import logging

def extract_text(file_path: str, doc_type: str):
    """
    Extract text from PDF or .docx files and return structured JSON data.
    
    Args:
        file_path (str): Path to the document (PDF or .docx).
        doc_type (str): Type of document ('resume' or 'job').
    
    Returns:
        dict: JSON data with doc_id, type, text, and word_count, or None if error.
    """
    # Validate file
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    
    # Validate file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 5:
        logging.error(f"File too large: {file_size_mb:.2f}MB (max 5MB)")
        return None
    
    logging.info(f"Processing {doc_type} file: {file_path}")
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        # Initialize text and metadata
        text = ""  
        word_count = 0

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower() 

        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + " "

        elif file_ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + " "

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Clean text: remove extra whitespace, newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Calculate word count for Resume Quality Score
        word_count = len(text.split())
        
        # Return structured JSON
        logging.info(f"Text extraction successful for {file_path}. Word count: {word_count}")
        return {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "text": text,
            "word_count": word_count
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None
    

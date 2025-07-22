# Contextual Job Description Matcher
The `Contextual Job Description Matcher` robust, modular backend system for automated matching of resumes to job descriptions using advanced skill extraction, semantic embeddings, and Retrieval-Augmented Generation (RAG) techniques.  
This project leverages Google Gemini for skill extraction and semantic similarity, and ChromaDB for efficient vector search.


## Features

- **Automated Resume & JD Parsing:** Extracts and cleans text from PDF and DOCX files.
- **Skill Extraction:** Uses Google Gemini to extract and normalize technical and soft skills from documents.
- **Semantic Embedding & Vector Search:** Generates embeddings for skills and performs similarity search using ChromaDB.
- **RAG Pipeline:** Matches resumes to job descriptions, computes match scores, missing skills, and provides improvement suggestions.
- **Metrics Calculation:** Computes relevance and completeness scores for each resume.
- **Structured Logging:** All processing steps and errors are logged for traceability.
- **Extensible & Modular:** Easily add new extractors, scoring methods, or data sources.

---

## Architecture Overview
```
              Resume (.pdf /.docx)        Job Description (.pdf /.docx)
                      |                              |
                      v                              v
               ┌────────────────────────────────────────────┐
               │              Document Processor             │
               │ Extracts plain text from uploaded documents │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │              Skill Extractor               │
               │ Uses NLP (SpaCy + rule-based patterns)     │
               │ to identify domain-specific skills         │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │              Data Formatter                │
               │ Cleans, normalizes, and structures input   │
               │ data for embedding and LLM consumption     │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │           Embedding Generator              │
               │ Converts formatted resume and JD into      │
               │ dense vector representations using Gemini  │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │               Vector Store                 │
               │ Stores embeddings for fast retrieval       │
               │ (e.g., Chroma or FAISS)                    │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │             RAG Pipeline (optional)        │
               │ Performs semantic retrieval of relevant    │
               │ context to enhance prompt inputs           │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │         Gemini LLM Evaluation Module       │
               │ Analyzes alignment between resume and JD   │
               │ Generates match metrics and feedback       │
               └────────────────────────────────────────────┘
                                |
                                v
               ┌────────────────────────────────────────────┐
               │              Output Generator              │
               │ Returns structured JSON containing:        │
               │ - Matched Skills                           │
               │ - Missing / Gap Skills                     │
               │ - Matching Score (0–100)                   │
               │ - Resume Improvement Suggestions           │
               │ - Role Fit Summary (Strong/Moderate/Weak)  │
               └────────────────────────────────────────────┘

```

---

## How It Works

1. **Document Processing:**  
   - Extracts and cleans text from resumes and job descriptions.
2. **Skill Extraction:**  
   - Uses Gemini to extract and normalize skills, mapping synonyms to canonical names from `skills.json`.
3. **Data Formatting:**  
   - Merges extracted data and skills into a unified JSON structure.
4. **Dataset Integration:**  
   - Integrates all resumes and JDs, computes metrics (relevance, completeness).
5. **Embedding & Vector Search:**  
   - Generates semantic embeddings for skills and stores them in ChromaDB.
6. **RAG Pipeline:**  
   - Matches each resume to job descriptions, computes match scores, missing skills, and provides suggestions.
7. **Output:**  
   - Saves processed results as JSON files in `data/processed/`.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/BenGJ10/Contextual-Job-Description-Matcher.git>
   cd Job\ Description\ Matcher
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory with your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_gemini_api_key
     ```

---

## Configuration

- **Skills Configuration:**  
  - Edit `config/skills.json` to update the canonical list of technical and soft skills.
- **Logging:**  
  - Logs are saved in the `logs/` directory with timestamped filenames.

---

## Usage

### **Processing Resumes and Job Descriptions**

1. **Add your resumes and job descriptions:**
   - Place resume files in `data/resumes/`
   - Place job description files in `data/jobs/`

2. **Run the RAG pipeline:**
   ```bash
   python src/tests/test_rag_pipeline.py
   ```
   - This will process all resumes and job descriptions, generate embeddings, perform matching, and save results in `data/processed/`.

3. **Check the output:**
   - Processed JSON files for each resume will be available in `data/processed/`, including:
     - Extracted skills
     - Metrics (relevance and completeness scores)
     - Job matches with scores, missing skills, and suggestions

---

## Project Structure

```
Job Description Matcher/
├── config/
│   └── skills.json
├── data/
│   ├── resumes/
│   ├── jobs/
│   └── processed/
├── logs/
├── src/
│   ├── backend/
│   │   ├── processing/
│   │   │   ├── document_processor.py
│   │   │   ├── skill_extractor.py
│   │   │   └── dataset_integrator.py
│   │   ├── rag/
│   │   │   └── rag_pipeline.py
│   │   └── utils/
│   │       ├── data_formatter.py
│   │       └── logger.py
│   └── tests/
│       └── test_rag_pipeline.py
├── requirements.txt
└── README.md
```

---

## Testing

- **Unit and integration tests** are available in `src/tests/`.
- To run all tests:
  ```bash
  python -m unittest discover src/tests/
  ```

---

## Logging

- All logs are saved in the `logs/` directory with detailed timestamps and context.
- Errors, warnings, and info messages are logged for traceability.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Google Gemini](https://ai.google.com/gemini/) for advanced language and embedding models.
- [ChromaDB](https://www.trychroma.com/) for vector database and similarity search.
- [LangChain](https://www.langchain.com/) for prompt and pipeline utilities.


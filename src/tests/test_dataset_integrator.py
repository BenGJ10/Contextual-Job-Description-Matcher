import os
import sys
from docx import Document

# Add root path so src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.backend.processing.dataset_integrator import process_dataset


def create_docx(path, text):
    """Helper function to create .docx files from raw text."""
    doc = Document()
    for line in text.strip().split("\n"):
        doc.add_paragraph(line.strip())
    doc.save(path)


if __name__ == "__main__":
    # Create mock job descriptions
    job_dir = "data/jobs"
    os.makedirs(job_dir, exist_ok=True)

    job_descriptions = {
        "job_1.docx": """
            Job Title: Data Analyst
            Skills: Python, SQL, Pandas, NumPy, Data Visualization, Tableau, Statistics
            Responsibilities: Analyze large datasets, create visualizations, and derive insights.
        """,
        "job_2.docx": """
            Job Title: Junior Data Analyst
            Skills: Python, Excel, SQL, Data Cleaning, R
            Responsibilities: Support data analysis projects, clean datasets, and assist with reporting.
        """,
        "job_3.docx": """
            Job Title: Data Scientist
            Skills: Python, Machine Learning, Scikit-learn, SQL, Deep Learning, Pandas, NumPy, Power BI
            Responsibilities: Build predictive models, analyze complex datasets, and deploy ML solutions.
        """
    }

    for filename, content in job_descriptions.items():
        create_docx(os.path.join(job_dir, filename), content)

    # Process resumes against jobs
    results = process_dataset(resume_dir="data/resumes", job_dir=job_dir)

    for result in results:
        if result:
            print(f"\nProcessed data for resume (ID: {result['doc_id']}):")
            print(result)
        else:
            print(f"\nFailed to process resume")
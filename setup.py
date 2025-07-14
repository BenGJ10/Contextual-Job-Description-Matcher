from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    requirement_lst: List[str] = []
    try:
        with open("requirements.txt") as f:
            lines = f.readlines()
            for line in lines:
                requirement = line.strip()
                # Ignore empty lines and editable installs
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("The requirements.txt file was not found!")
    return requirement_lst

setup(
    name="JobDescriptionMatcher",
    version="0.1.0",
    description="End-to-end system to match resumes with job descriptions using RAG, LangChain, and AWS",
    author="Ben Gregory John",
    author_email = "bengj1015@gmail.com",
    packages=find_packages(where = "src"),
    package_dir = {"": "src"},
    include_package_data = True,
    install_requires = [],
    python_requires = ">=3.10",
)
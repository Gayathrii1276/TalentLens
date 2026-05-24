# TalentLens – AI-Powered Resume Analyzer & Recruiter Assistant

TalentLens is an AI-powered recruitment and resume intelligence platform built using Streamlit, NLP, semantic similarity, and Generative AI.

The platform helps recruiters shortlist candidates efficiently and enables job seekers to optimize their resumes against job descriptions using ATS-style analysis.

---

# Features

## AI Resume vs Job Description Matching

* Upload resumes in PDF format
* Compare resumes against job descriptions
* Generate ATS match scores instantly
* Identify missing skills and keywords

## Semantic Resume Analysis

* Uses sentence embeddings and cosine similarity
* Evaluates resumes beyond exact keyword matching
* Provides more accurate candidate ranking

## GitHub Profile Intelligence

* Detects GitHub profiles from resumes automatically
* Analyzes repositories and coding activity
* Evaluates developer presence and technical depth
* Generates GitHub-based candidate insights

## Recruiter Dashboard

* Rank multiple candidates automatically
* Compare resumes side-by-side
* Display ATS score, semantic score, and keyword score
* Generate recruiter-friendly summaries

## AI-Powered Insights

* Resume summary generation
* Skill gap identification
* AI-generated interview questions
* Candidate evaluation assistance

## Modern UI/UX

* Custom dark-themed Streamlit interface
* Interactive cards and analytics
* Clean recruiter-focused dashboard

---

# Tech Stack

## Frontend

* Streamlit
* Custom CSS

## Backend & AI

* Python
* Groq API
* Sentence Transformers
* TF-IDF Vectorization
* Cosine Similarity
* NLP-based Resume Parsing

## Libraries Used

* PyPDF2
* NumPy
* Requests
* BeautifulSoup4
* GitPython
* Radon
* python-dotenv

---

# Project Architecture

```text
TalentLens/
│
├── app.py                  # Main Streamlit application
├── github_analyzer.py      # GitHub profile & repository analysis
├── requirements.txt        # Project dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Gayathrii1276/TalentLens.git
cd TalentLens
```

---

## 2. Create Virtual Environment

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Setup Environment Variables

Create a `.env` file in the root directory.

```env
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
```

---

# Running the Application

```bash
streamlit run app.py
```

The application will start at:

```text
http://localhost:8501
```

---

# How TalentLens Works

## Step 1 – Upload Resume

Users upload resumes in PDF format.

## Step 2 – Add Job Description

Paste the target job description.

## Step 3 – AI Processing

TalentLens performs:

* Resume parsing
* Keyword extraction
* Semantic similarity analysis
* ATS scoring
* GitHub analysis

## Step 4 – Results Generation

The platform generates:

* ATS match percentage
* Missing skills
* Semantic score
* Candidate summaries
* Interview questions

---

# ATS Scoring Logic

TalentLens combines multiple evaluation methods:

| Metric          | Description              |
| --------------- | ------------------------ |
| Keyword Score   | Skill & keyword matching |
| Semantic Score  | Meaning-based similarity |
| LLM Score       | AI-generated evaluation  |
| Final ATS Score | Weighted combined score  |

---

# GitHub Analyzer

The integrated GitHub analyzer:

* Extracts GitHub links from resumes
* Reads public repositories
* Analyzes coding activity
* Detects technologies used
* Evaluates repository quality
* Generates developer insights

This helps recruiters assess real-world technical skills beyond resumes.

---

# Screenshots

## Dashboard

Add your application screenshots here.

```text
assets/dashboard.png
```

## Resume Analysis

```text
assets/resume-analysis.png
```

## Candidate Ranking

```text
assets/candidate-ranking.png
```

---

# Future Improvements

* Resume database support
* Authentication system
* Recruiter analytics dashboard
* Multi-format resume support
* Advanced AI interview assistant
* Deployment on cloud platforms
* Real-time recruiter collaboration

---

# Deployment

## Streamlit Cloud

1. Push project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repository
4. Add environment variables
5. Deploy application

---

# Security Notes

* Never expose API keys publicly
* Use `.env` for secrets
* Keep `.env` inside `.gitignore`
* Use GitHub Push Protection

---

# Author

## Kommineni Gayathri

AI/ML Enthusiast | Python Developer | Generative AI Builder

GitHub:
[https://github.com/Gayathrii1276](https://github.com/Gayathrii1276)

LinkedIn:
www.linkedin.com/in/gayathri-kommineni-7b498b269
---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# Why TalentLens?

TalentLens was built to bridge the gap between recruiters and candidates using AI-driven resume intelligence.

Traditional ATS systems rely heavily on keyword matching and often fail to identify strong candidates.

TalentLens combines:

* Semantic understanding
* AI reasoning
* GitHub portfolio analysis
* Recruiter-focused insights

to create a smarter and more human-centric recruitment assistant.

import os
import streamlit as st
import json
import PyPDF2
import numpy as np

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import faiss
except Exception:
    faiss = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    def cosine_similarity(a, b):
        return np.zeros((1, 1))

from github_analyzer import extract_github_url, analyze_github_profile
import github_analyzer as _gh_mod


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentLens · AI ATS", page_icon="🔍", layout="wide")

# ── Global CSS (fonts, background, native widget overrides only) ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] { background: #0f0f18 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #3d3d6b; border-radius: 2px; }

/* Inputs */
textarea, .stTextArea textarea {
    background: #11111e !important;
    border: 1px solid #22223d !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #11111e !important;
    border: 1.5px dashed #2a2a50 !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6366f1 !important;
    background: #13132a !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.8rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #11111e !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6e6e9e !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0d0d18 !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #a5b4fc !important;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: #11111e !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: #11111e !important;
    border: 1px solid #22223d !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    background: #11111e !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Headings */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #e8e6f0 !important;
}

/* Spinner */
[data-testid="stSpinner"] p {
    color: #a5b4fc !important;
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ── Reusable inline-HTML helpers (self-contained, always closed) ──────────────
def _pill(text, bg, border, color):
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'font-size:0.72rem;font-weight:500;margin:2px;'
            f'background:{bg};border:1px solid {border};color:{color};">{text}</span>')

def skill_pill(text):
    return _pill(text, "rgba(99,102,241,0.12)", "rgba(99,102,241,0.25)", "#a5b4fc")

def match_pill(text):
    return _pill(f"✓ {text}", "rgba(52,211,153,0.15)", "rgba(52,211,153,0.35)", "#34d399")

def unmatch_pill(text):
    return _pill(text, "rgba(107,114,128,0.1)", "rgba(107,114,128,0.2)", "#6b7280")

def missing_pill(text):
    return _pill(f"✗ {text}", "rgba(239,68,68,0.1)", "rgba(239,68,68,0.25)", "#fca5a5")

def pill_row(pills_html):
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin:6px 0;">{pills_html}</div>'

def section_header(title):
    st.markdown(
        f'<p style="font-size:0.68rem;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;color:#818cf8;margin:0 0 0.6rem 0;">{title}</p>',
        unsafe_allow_html=True,
    )

def divider():
    st.markdown(
        '<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,#2a2a50,transparent);margin:1.5rem 0;">',
        unsafe_allow_html=True,
    )

def score_color(v):
    return "#34d399" if v >= 80 else "#fbbf24" if v >= 60 else "#f87171"


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f0f1a 0%,#12122a 50%,#0a0a0f 100%);
     border:1px solid #1e1e3f;border-radius:20px;padding:2.5rem 3rem;margin-bottom:2rem;">
  <div style="display:inline-block;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
       color:#a5b4fc;font-size:0.7rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;
       padding:4px 12px;border-radius:20px;margin-bottom:1rem;">AI-Powered Recruitment</div>
  <h1 style="font-family:'Syne',sans-serif!important;font-size:2.6rem;font-weight:800;margin:0 0 0.4rem 0;
       background:linear-gradient(90deg,#a5b4fc 0%,#818cf8 40%,#34d399 100%);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
       TalentLens</h1>
  <p style="color:#9998b8;font-size:1rem;font-weight:300;margin:0;">
       Multi-signal ATS scoring · GitHub intelligence · AI interview prep · Recruiter chatbot</p>
</div>
""", unsafe_allow_html=True)


# ── API setup ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "Your-GROQ-API-Key-Here")
if not GROQ_API_KEY:
    st.error("⚠️  GROQ_API_KEY environment variable not set.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

if os.environ.get("GITHUB_TOKEN"):
    _gh_mod.GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]


# ── Embedding model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()


# ── Core functions ────────────────────────────────────────────────────────────
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def chunk_text(text, chunk_size=250):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def keyword_score(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 150
    return min(round(score, 2), 100)


def semantic_score(resume, jd):
    emb_r = embedding_model.encode([resume])
    emb_j = embedding_model.encode([jd])
    score = cosine_similarity(emb_r, emb_j)[0][0] * 130
    return min(round(score, 2), 100)


def llm_evaluation(resume, jd):
    prompt = f"""You are a smart ATS evaluator.
Compare the resume with the job description.
IMPORTANT: Mention ONLY important missing skills. Maximum 3 missing skills.
Only include skills clearly absent in the resume. Do NOT invent technologies.
Keep summary short and recruiter-friendly.
Return ONLY valid JSON (no markdown, no explanation).
Format:
{{
    "llm_score": 0-100,
    "missing_skills": [],
    "profile_summary": ""
}}
JOB DESCRIPTION:\n{jd}\nRESUME:\n{resume[:4000]}"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an ATS evaluator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2, max_tokens=300,
    )
    output = response.choices[0].message.content.strip()
    output = output.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(output)
    except Exception:
        return {"llm_score": 50, "missing_skills": [], "profile_summary": "Unable to evaluate resume"}


def calculate_ats(resume, jd, github_score=0):
    k = keyword_score(resume, jd)
    s = semantic_score(resume, jd)
    llm = llm_evaluation(resume, jd)
    final = 0.25 * k + 0.30 * s + 0.25 * llm["llm_score"] + 0.20 * github_score
    return {
        "final_score": round(final, 2),
        "keyword_score": k,
        "semantic_score": s,
        "llm_score": llm["llm_score"],
        "missing_skills": llm["missing_skills"],
        "summary": llm["profile_summary"],
    }


def generate_questions(resume, jd):
    prompt = f"""You are a senior technical recruiter.
Generate EXACTLY 7 interview questions for this candidate.
Rules:
- Relevant to the job description
- Focus on technical skills, projects, and problem solving
- Return ONLY a numbered list (1 to 7), no other text
- Each question should be short and clear
Structure: 1-3 Technical, 4-5 Project/experience, 6-7 Problem solving/behavioral
Job Description:\n{jd}\nCandidate Resume:\n{resume[:2000]}"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def build_vector_store(resumes, jd):
    texts, metadata = [], []
    for chunk in chunk_text(jd):
        texts.append(chunk); metadata.append("JOB_DESCRIPTION")
    for name, resume in resumes.items():
        for chunk in chunk_text(resume):
            texts.append(chunk); metadata.append(name)
    embeddings = embedding_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts, metadata


def retrieve_context(question, index, texts, metadata):
    q_emb = embedding_model.encode([question])
    _, indices = index.search(np.array(q_emb), 4)
    return "".join(f"\nSource: {metadata[i]}\n{texts[i]}\n" for i in indices[0])


def recruiter_chatbot(question, index, texts, metadata, jd):
    context = retrieve_context(question, index, texts, metadata)
    prompt = f"""You are an AI recruitment assistant.
Use the resume context and job description to answer the recruiter's question.
Job Description:\n{jd}\nResume Context:\n{context}\nRecruiter Question:\n{question}"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=400,
    )
    return response.choices[0].message.content


# ── INPUT SECTION ─────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([1.1, 1], gap="large")

with col_jd:
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin-bottom:0.4rem;">Job Description</p>', unsafe_allow_html=True)
    jd = st.text_area(label="jd_input", label_visibility="collapsed",
                      placeholder="Paste the full job description here…", height=220)

with col_up:
    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin-bottom:0.4rem;">Resume PDFs</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(label="upload_resumes", label_visibility="collapsed",
                                      type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")

st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 5])
with run_col:
    analyze_btn = st.button("🔍  Analyze Resumes", use_container_width=True)


# ── ANALYSIS ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not jd.strip():
        st.warning("Please paste a job description first.")
        st.stop()
    if not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
        st.stop()

    results, resumes = [], {}
    progress = st.progress(0, text="Analyzing candidates…")

    for idx, file in enumerate(uploaded_files):
        progress.progress(idx / len(uploaded_files), text=f"Processing {file.name}…")
        text = extract_text_from_pdf(file)
        resumes[file.name] = text

        file.seek(0)
        github_url = extract_github_url(text, pdf_file=file)
        github_report, github_score = None, 0

        with st.expander(f"🔍 GitHub debug — {file.name}", expanded=False):
            if github_url:
                st.success(f"✅ Detected GitHub URL: `{github_url}`")
            else:
                st.error("❌ No GitHub URL found in text or annotations.")
                st.code(text[:1500].replace('\n', ' '), language=None)

        if github_url:
            with st.spinner(f"🐙 Fetching GitHub profile for {file.name}…"):
                try:
                    github_report = analyze_github_profile(github_url, jd, groq_client=client)
                    github_score = github_report.get("final_github_score", 0) if github_report else 0
                except Exception as e:
                    st.warning(f"GitHub analysis failed for {file.name}: {e}")

        ats = calculate_ats(text, jd, github_score)
        questions = generate_questions(text, jd)
        results.append({
            "Candidate": file.name.replace(".pdf", ""),
            "ATS Score": ats["final_score"],
            "Keyword Score": ats["keyword_score"],
            "Semantic Score": ats["semantic_score"],
            "LLM Score": ats["llm_score"],
            "GitHub Score": github_score,
            "Missing Skills": ats.get("missing_skills", []),
            "Summary": ats.get("summary", ""),
            "Questions": questions,
            "GitHub Report": github_report,
        })

    progress.progress(1.0, text="Done!")
    progress.empty()
    st.session_state["results"] = sorted(results, key=lambda x: x["ATS Score"], reverse=True)
    st.session_state["resumes"] = resumes
    st.session_state["jd"] = jd


# ── DISPLAY RESULTS ───────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    jd = st.session_state.get("jd", "")

    divider()

    # ── Leaderboard ──
    st.markdown('<p style="font-size:0.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin-bottom:0.8rem;">Candidate Rankings</p>', unsafe_allow_html=True)
    table_data = [
        {"Rank": f"#{i+1}", "Candidate": r["Candidate"],
         "ATS Score": r["ATS Score"], "Keyword": r["Keyword Score"],
         "Semantic": r["Semantic Score"], "LLM": r["LLM Score"], "GitHub": r["GitHub Score"]}
        for i, r in enumerate(results)
    ]
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    divider()
    st.markdown('<p style="font-size:0.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin-bottom:1rem;">Detailed Reports</p>', unsafe_allow_html=True)

    for rank, r in enumerate(results, 1):
        ats_color = score_color(r["ATS Score"])

        # ── Candidate header ──────────────────────────────────────────────────
        col_rank, col_name, col_score = st.columns([0.08, 0.72, 0.2])
        with col_rank:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;'
                f'font-family:Syne,sans-serif;font-weight:800;font-size:1rem;'
                f'width:42px;height:42px;border-radius:10px;display:flex;align-items:center;'
                f'justify-content:center;box-shadow:0 4px 14px rgba(99,102,241,0.35);">#{rank}</div>',
                unsafe_allow_html=True,
            )
        with col_name:
            st.markdown(
                f'<p style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;'
                f'color:#e8e6f0;margin:0 0 2px 0;">{r["Candidate"]}</p>'
                f'<p style="font-size:0.78rem;color:#6e6e9e;margin:0;">Resume + GitHub Analysis</p>',
                unsafe_allow_html=True,
            )
        with col_score:
            st.markdown(
                f'<div style="text-align:right;">'
                f'<div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;'
                f'line-height:1;color:{ats_color};">{r["ATS Score"]}</div>'
                f'<div style="font-size:0.72rem;color:#6e6e9e;">ATS Score / 100</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ATS Breakdown ─────────────────────────────────────────────────────
        with st.expander("📊 ATS Breakdown", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                (m1, "Keyword", r["Keyword Score"], "#6366f1"),
                (m2, "Semantic", r["Semantic Score"], "#8b5cf6"),
                (m3, "LLM Score", r["LLM Score"], "#06b6d4"),
                (m4, "GitHub", r["GitHub Score"], "#34d399"),
            ]
            for col, label, val, color in metrics:
                with col:
                    st.markdown(
                        f'<div style="background:#11111e;border:1px solid #1e1e3a;border-top:2px solid {color};'
                        f'border-radius:12px;padding:1rem;text-align:center;">'
                        f'<div style="font-size:0.62rem;font-weight:600;letter-spacing:1.8px;'
                        f'text-transform:uppercase;color:#6e6e9e;margin-bottom:6px;">{label}</div>'
                        f'<div style="font-family:Syne,sans-serif;font-size:1.9rem;font-weight:800;color:{color};line-height:1;">{val}</div>'
                        f'<div style="background:#1a1a2e;border-radius:20px;height:4px;margin-top:10px;overflow:hidden;">'
                        f'<div style="height:100%;width:{min(float(val),100)}%;border-radius:20px;background:{color};"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            if r["Summary"]:
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"💡 {r['Summary']}")

            if r["Missing Skills"]:
                st.markdown("<br>", unsafe_allow_html=True)
                section_header("Missing Skills")
                pills = "".join(missing_pill(s) for s in r["Missing Skills"])
                st.markdown(pill_row(pills), unsafe_allow_html=True)

        # ── GitHub Intelligence ───────────────────────────────────────────────
        gh = r.get("GitHub Report")
        with st.expander("🐙 GitHub Intelligence", expanded=True):
            if not gh:
                st.warning("No GitHub profile detected in this resume.")
            else:
                jd_text = jd if jd else ""
                jd_lower = jd_text.lower()
                detected_skills = gh.get("detected_skills", [])
                jd_matched   = [s for s in detected_skills if s.lower() in jd_lower]
                jd_unmatched = [s for s in detected_skills if s.lower() not in jd_lower]
                jd_match_pct = int(len(jd_matched) / max(len(detected_skills), 1) * 100)
                score_val    = gh["final_github_score"]
                gh_color     = score_color(score_val)

                # Profile header row
                gh_col1, gh_col2, gh_col3, gh_col4 = st.columns([0.5, 0.5, 0.5, 0.5])
                with gh_col1:
                    st.markdown(f'<p style="margin:0;font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:#c7d2fe;">🐙 @{gh.get("username","")}</p>'
                                f'<a href="{gh["github_url"]}" target="_blank" style="font-size:0.75rem;color:#6366f1;">{gh["github_url"]}</a>',
                                unsafe_allow_html=True)
                    if gh.get("bio"):
                        st.caption(gh["bio"])
                with gh_col2:
                    st.metric("GitHub Score", f"{score_val} / 100")
                with gh_col3:
                    st.metric("Repos", gh.get("total_repos", 0))
                with gh_col4:
                    st.metric("Followers", gh.get("followers", 0))

                st.markdown("<br>", unsafe_allow_html=True)

                # JD Stack Match
                section_header(f"JD Stack Match — {jd_match_pct}%")
                st.progress(jd_match_pct / 100)
                match_pills   = "".join(match_pill(s)   for s in jd_matched)
                unmatch_pills = "".join(unmatch_pill(s) for s in jd_unmatched)
                st.markdown(pill_row(match_pills + unmatch_pills), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # All detected skills
                section_header("All Detected Skills")
                all_pills = "".join(skill_pill(s) for s in detected_skills)
                st.markdown(pill_row(all_pills) if all_pills else "*None detected*", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Top repositories
                section_header("Top Repositories")
                for repo in gh.get("top_repositories", []):
                    repo_stack       = repo.get("tech_stack", [])
                    repo_jd_matched  = [t for t in repo_stack if t.lower() in jd_lower]
                    match_ratio      = int(len(repo_jd_matched) / max(len(repo_stack), 1) * 100) if repo_stack else 0
                    badge_color      = "#34d399" if match_ratio >= 60 else "#fbbf24" if match_ratio >= 30 else "#6b7280"

                    with st.container():
                        r_col1, r_col2 = st.columns([0.8, 0.2])
                        with r_col1:
                            st.markdown(
                                f'<p style="font-family:Syne,sans-serif;font-weight:600;color:#a5b4fc;'
                                f'font-size:0.95rem;margin:0;">📦 {repo["name"]}</p>'
                                f'<p style="color:#7878a8;font-size:0.78rem;margin:2px 0 6px 0;">{repo["description"]}</p>',
                                unsafe_allow_html=True,
                            )
                            tech_pills = "".join(
                                match_pill(t) if t in repo_jd_matched else skill_pill(t)
                                for t in repo_stack
                            )
                            st.markdown(pill_row(tech_pills), unsafe_allow_html=True)
                        with r_col2:
                            st.markdown(
                                f'<div style="text-align:right;padding-top:4px;">'
                                f'<span style="font-size:0.75rem;font-weight:700;color:{badge_color};">JD {match_ratio}%</span><br>'
                                f'<span style="font-size:0.72rem;color:#5a5a8a;">⭐ {repo["stars"]} &nbsp; 🍴 {repo.get("forks",0)}</span><br>'
                                f'<a href="{repo["url"]}" target="_blank" style="font-size:0.72rem;color:#6366f1;">View →</a></div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown('<hr style="border:none;border-top:1px solid #1e1e3a;margin:8px 0;">', unsafe_allow_html=True)

        # ── Interview Questions ───────────────────────────────────────────────
        with st.expander("💬 Interview Questions", expanded=True):
            if r["Questions"]:
                for line in r["Questions"].strip().split("\n"):
                    line = line.strip()
                    if line:
                        st.markdown(
                            f'<div style="background:#0f0f1a;border:1px solid #1a1a30;'
                            f'border-left:3px solid #6366f1;border-radius:0 10px 10px 0;'
                            f'padding:0.8rem 1rem;margin-bottom:0.5rem;'
                            f'font-size:0.88rem;color:#c8c8e8;line-height:1.5;">{line}</div>',
                            unsafe_allow_html=True,
                        )

        divider()


# ── CHATBOT ───────────────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#6366f1;margin-bottom:0.3rem;">Recruiter Intelligence</p>', unsafe_allow_html=True)
st.markdown("## 🤖 Ask the Chatbot")
st.caption("Query across all uploaded resumes and the job description. Build the knowledge base first.")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

resumes_available = "resumes" in st.session_state and st.session_state["resumes"]
_jd_for_chat = st.session_state.get("jd", "")

if resumes_available and _jd_for_chat:
    build_col, _ = st.columns([1, 4])
    with build_col:
        if st.button("⚡  Build Knowledge Base", use_container_width=True):
            with st.spinner("Building vector database…"):
                index, texts, metadata = build_vector_store(st.session_state["resumes"], _jd_for_chat)
                st.session_state.vector_db = (index, texts, metadata)
            st.success("✅ Chatbot is ready. Ask away!")
else:
    st.info("Run an analysis first to enable the chatbot.", icon="ℹ️")

if st.session_state.vector_db:
    question = st.chat_input("Ask about candidates, skills, fit, red flags…")
    if question:
        index, texts, metadata = st.session_state.vector_db
        answer = recruiter_chatbot(question, index, texts, metadata, _jd_for_chat)
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
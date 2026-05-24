import re
import math
import base64
import requests
import PyPDF2

# ── Token: read from env (set via Streamlit secrets or os.environ) ────────────
import os
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

def _make_headers():
    """Build headers — works with or without a token (unauthenticated = 60 req/hr)."""
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN and GITHUB_TOKEN != "your_github_token_here":
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

RESERVED_PATHS = {
    'features','pricing','about','login','signup','join','contact',
    'explore','marketplace','topics','trending','collections','events',
    'sponsors','readme','issues','pulls','wiki','blob','tree','commit',
    'releases','tags','actions','projects','security','pulse','graphs',
    'settings','notifications','orgs','organizations','apps','integrations',
    'new','search','dashboard','account','enterprise','business',
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF annotation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_urls_from_pdf_annotations(pdf_file) -> list:
    urls = []
    try:
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            if '/Annots' not in page:
                continue
            annots = page['/Annots']
            if annots is None:
                continue
            for annot_ref in annots:
                try:
                    obj = annot_ref.get_object()
                    if obj.get('/Subtype') != '/Link':
                        continue
                    action = obj.get('/A')
                    if action is None:
                        continue
                    if hasattr(action, 'get_object'):
                        action = action.get_object()
                    if '/URI' not in action:
                        continue
                    uri = action['/URI']
                    if hasattr(uri, 'get_object'):
                        uri = uri.get_object()
                    if isinstance(uri, bytes):
                        uri = uri.decode('utf-8', errors='ignore')
                    urls.append(str(uri).strip())
                except Exception:
                    continue
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
    except Exception:
        pass
    return urls


def _parse_github_username(raw: str):
    if not raw:
        return None
    raw = raw.split('?')[0].split('#')[0].strip().rstrip('/')
    m = re.search(
        r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9][A-Za-z0-9\-]{0,38})',
        raw, re.IGNORECASE
    )
    if m:
        username = m.group(1).rstrip('.-,;)')
        if username.lower() not in RESERVED_PATHS:
            return username, f"https://github.com/{username}"
    return None


def extract_github_url(text: str, pdf_file=None) -> str | None:
    if pdf_file is not None:
        for url in extract_urls_from_pdf_annotations(pdf_file):
            parsed = _parse_github_username(url)
            if parsed:
                return parsed[1]

    text = (text
            .replace('\u2013', '-').replace('\u2014', '-')
            .replace('\u2018', "'").replace('\u2019', "'")
            .replace('\u201c', '"').replace('\u201d', '"'))
    text = re.sub(r'(github\.com)\s*/\s*', r'\1/', text, flags=re.IGNORECASE)
    text = text.replace('\n', ' ').replace('\r', ' ')

    url_pattern = (
        r'(?:https?://)?(?:www\.)?github\.com/'
        r'([A-Za-z0-9][A-Za-z0-9\-]{0,38}[A-Za-z0-9]?)'
        r'(?=[/\s,;.)\]>"\']|$)'
    )
    for m in re.finditer(url_pattern, text, re.IGNORECASE):
        username = m.group(1).rstrip('.-')
        if username.lower() not in RESERVED_PATHS and len(username) >= 1:
            return f"https://github.com/{username}"

    label_pattern = (
        r'[Gg]it\s*[Hh]ub'
        r'(?:\s*(?:[Pp]rofile|[Pp]age|[Uu]rl|[Ll]ink|[Aa]ccount))?'
        r'\s*[:\|]\s*'
        r'([A-Za-z0-9][A-Za-z0-9\-]{0,38}[A-Za-z0-9]?)'
    )
    for m in re.finditer(label_pattern, text):
        username = m.group(1).strip()
        if username.lower() not in RESERVED_PATHS and '.' not in username:
            return f"https://github.com/{username}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GitHub API helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_username(github_url):
    return github_url.rstrip('/').split('/')[-1]


def fetch_repositories(username):
    url = f'https://api.github.com/users/{username}/repos?per_page=100&sort=updated'
    try:
        r = requests.get(url, headers=_make_headers(), timeout=10)
        if r.status_code == 401:
            print("GitHub API: Unauthorized — check your GITHUB_TOKEN.")
            return []
        if r.status_code == 404:
            print(f"GitHub user '{username}' not found.")
            return []
        if r.status_code != 200:
            print(f"GitHub API error {r.status_code}: {r.text[:200]}")
            return []
        return r.json()
    except Exception as e:
        print(f"fetch_repositories error: {e}")
        return []


def fetch_user_profile(username):
    try:
        r = requests.get(f'https://api.github.com/users/{username}',
                         headers=_make_headers(), timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# README-first stack detection via GitHub Contents API
# ─────────────────────────────────────────────────────────────────────────────

# Keyword → canonical skill label
TECH_MAP = {
    # Python web
    'fastapi': 'FastAPI', 'django': 'Django', 'flask': 'Flask',
    'tornado': 'Tornado', 'aiohttp': 'aiohttp', 'starlette': 'Starlette',
    # JS / frontend
    'react': 'React', 'vue': 'Vue.js', 'angular': 'Angular',
    'next.js': 'Next.js', 'next': 'Next.js', 'nuxt': 'Nuxt.js',
    'svelte': 'Svelte', 'vite': 'Vite', 'webpack': 'Webpack',
    'tailwind': 'TailwindCSS', 'bootstrap': 'Bootstrap',
    # Node / backend JS
    'node.js': 'Node.js', 'node': 'Node.js', 'express': 'Express',
    'nestjs': 'NestJS', 'graphql': 'GraphQL', 'prisma': 'Prisma',
    # Databases
    'mongodb': 'MongoDB', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL',
    'mysql': 'MySQL', 'sqlite': 'SQLite', 'redis': 'Redis',
    'elasticsearch': 'Elasticsearch', 'cassandra': 'Cassandra',
    'supabase': 'Supabase', 'firebase': 'Firebase', 'dynamodb': 'DynamoDB',
    # DevOps / infra
    'docker': 'Docker', 'kubernetes': 'Kubernetes', 'terraform': 'Terraform',
    'ansible': 'Ansible', 'nginx': 'Nginx', 'github actions': 'GitHub Actions',
    'ci/cd': 'CI/CD', 'jenkins': 'Jenkins',
    # Cloud
    'aws': 'AWS', 'gcp': 'GCP', 'azure': 'Azure',
    'vercel': 'Vercel', 'heroku': 'Heroku', 'netlify': 'Netlify',
    # AI / ML
    'langchain': 'LangChain', 'langgraph': 'LangGraph',
    'streamlit': 'Streamlit', 'gradio': 'Gradio',
    'tensorflow': 'TensorFlow', 'pytorch': 'PyTorch', 'torch': 'PyTorch',
    'keras': 'Keras', 'openai': 'OpenAI', 'anthropic': 'Anthropic',
    'transformers': 'HuggingFace Transformers', 'huggingface': 'HuggingFace',
    'hugging face': 'HuggingFace',
    'scikit-learn': 'scikit-learn', 'scikit': 'scikit-learn',
    'sklearn': 'scikit-learn',
    'pandas': 'Pandas', 'numpy': 'NumPy', 'opencv': 'OpenCV',
    'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost',
    'matplotlib': 'Matplotlib', 'seaborn': 'Seaborn', 'plotly': 'Plotly',
    'llm': 'LLM', 'rag': 'RAG', 'vector database': 'Vector DB',
    'pinecone': 'Pinecone', 'weaviate': 'Weaviate', 'chroma': 'ChromaDB',
    'faiss': 'FAISS', 'llamaindex': 'LlamaIndex', 'llama': 'LLaMA',
    # Backend misc
    'celery': 'Celery', 'spring': 'Spring Boot', 'grpc': 'gRPC',
    'kafka': 'Kafka', 'rabbitmq': 'RabbitMQ', 'websocket': 'WebSocket',
    'rest api': 'REST API', 'microservice': 'Microservices',
    # Languages (from README prose)
    'python': 'Python', 'javascript': 'JavaScript', 'typescript': 'TypeScript',
    'golang': 'Go', 'go ': 'Go', 'rust': 'Rust', 'java': 'Java',
    'kotlin': 'Kotlin', 'swift': 'Swift', 'c++': 'C++', 'c#': 'C#',
}

# All filenames to try fetching, in priority order
README_NAMES = ['README.md', 'readme.md', 'Readme.md', 'README.MD',
                'README.rst', 'README.txt', 'readme.rst']

STACK_FILES = [
    'requirements.txt',
    'package.json',
    'Dockerfile',
]


def fetch_raw_file(username, repo_name, filepath):
    """
    Fetch a file via GitHub Contents API and return its decoded text.
    Returns empty string on any error or missing file.
    """
    url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{filepath}"
    try:
        r = requests.get(url, headers=_make_headers(), timeout=10)
        if r.status_code != 200:
            return ""
        data = r.json()
        if isinstance(data, list):
            return ""  # it's a directory, not a file
        encoding = data.get('encoding', '')
        content = data.get('content', '')
        if encoding == 'base64':
            return base64.b64decode(content).decode('utf-8', errors='ignore')
        return content
    except Exception:
        return ""


def fetch_readme(username, repo_name):
    """Try every README filename variant until one works."""
    # First try the GitHub API's dedicated README endpoint (handles all variants)
    try:
        r = requests.get(
            f"https://api.github.com/repos/{username}/{repo_name}/readme",
            headers=_make_headers(), timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            content = data.get('content', '')
            if data.get('encoding') == 'base64':
                return base64.b64decode(content).decode('utf-8', errors='ignore')
    except Exception:
        pass
    # Fallback: try filenames manually
    for name in README_NAMES:
        text = fetch_raw_file(username, repo_name, name)
        if text:
            return text
    return ""


def keyword_extract_skills(text):
    """Fast keyword scan against TECH_MAP."""
    text_lower = text.lower()
    return list({label for kw, label in TECH_MAP.items() if kw in text_lower})


def llm_extract_skills_from_readme(readme_text, repo_name, groq_client=None):
    """
    Use LLM to read the README and extract tech skills intelligently.
    Falls back to keyword extraction if no LLM client provided.
    """
    if not readme_text:
        return []

    if groq_client is None:
        return keyword_extract_skills(readme_text)

    # Trim README to avoid token limits
    snippet = readme_text[:3000]
    prompt = f"""You are a technical recruiter assistant. Read this GitHub README for a project called "{repo_name}" and extract ALL technologies, frameworks, libraries, tools, languages, and platforms used or mentioned.

README:
---
{snippet}
---

Return ONLY a JSON array of strings, each being a specific technology name (e.g. ["Python", "FastAPI", "PostgreSQL", "Docker", "React"]).
No explanation, no markdown, just the JSON array. If nothing technical is found, return [].
"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        import json
        skills = json.loads(raw)
        if isinstance(skills, list):
            return [str(s).strip() for s in skills if s]
    except Exception as e:
        print(f"LLM skill extraction failed for {repo_name}: {e}")
    # Fallback to keyword scan
    return keyword_extract_skills(readme_text)


def detect_tech_stack_for_repo(username, repo_name, groq_client=None):
    """
    Full stack detection pipeline for a single repo:
    1. Fetch README → LLM extraction (rich, understands prose)
    2. Fetch stack files (requirements.txt, package.json, etc.) → keyword scan
    3. Merge and deduplicate
    """
    skills = set()

    # ── Step 1: README → LLM ──────────────────────────────────────────────────
    readme = fetch_readme(username, repo_name)
    if readme:
        llm_skills = llm_extract_skills_from_readme(readme, repo_name, groq_client)
        skills.update(llm_skills)
        # Also always keyword-scan the README as a safety net
        skills.update(keyword_extract_skills(readme))

    # ── Step 2: Dependency/config files → keyword scan ────────────────────────
    for filepath in STACK_FILES:
        content = fetch_raw_file(username, repo_name, filepath)
        if content:
            skills.update(keyword_extract_skills(content))

    return list(skills)


# ─────────────────────────────────────────────────────────────────────────────
# GitHub score calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_github_score(repos, user_profile, detected_skills, jd_text, top_repositories=None):
    """
    Score breakdown (100 pts total):
      • Project quality    40 pts  — per-repo stars + recency + description
      • JD stack match     35 pts  — weighted overlap of detected skills vs JD
      • Profile signals    15 pts  — followers, bio, blog/social
      • Breadth/activity   10 pts  — repo count + recent activity
    """
    from datetime import datetime, timezone
    score = 0.0
    jd_lower = jd_text.lower() if jd_text else ""
    now = datetime.now(timezone.utc)

    # ── 1. Project quality (40 pts) ──────────────────────────────────────────
    repo_score_total = 0.0
    for repo in repos:
        r_score = 0.0
        stars = repo.get('stargazers_count', 0)
        r_score += min(math.log1p(stars) / math.log1p(100), 1.0) * 4
        if repo.get('description'):
            r_score += 1.0
        updated = repo.get('updated_at')
        if updated:
            try:
                days_old = (now - datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ")
                            .replace(tzinfo=timezone.utc)).days
                r_score += max(0.0, 1.0 - days_old / 730) * 2
            except Exception:
                pass
        r_score += min(repo.get('forks_count', 0) * 0.2, 1.0)
        repo_score_total += r_score
    score += min(repo_score_total / (10 * 8) * 40, 40)

    # ── 2. JD stack match (35 pts) ───────────────────────────────────────────
    if jd_lower:
        if top_repositories:
            weighted_matched = 0.0
            weighted_total = 0.0
            for repo in top_repositories:
                stack = repo.get('tech_stack', [])
                if not stack:
                    continue
                weight = 1 + math.log1p(repo.get('stars', 0))
                matched = sum(1 for t in stack if t.lower() in jd_lower)
                weighted_matched += weight * (matched / len(stack))
                weighted_total += weight
            score += (weighted_matched / max(weighted_total, 1)) * 35
        elif detected_skills:
            matched = sum(1 for s in detected_skills if s.lower() in jd_lower)
            score += (matched / max(len(detected_skills), 1)) * 35
    else:
        score += min(len(detected_skills) * 2, 20)

    # ── 3. Profile signals (15 pts) ──────────────────────────────────────────
    if user_profile.get('bio'):
        score += 4
    score += min(user_profile.get('followers', 0) / 200, 1.0) * 8
    if user_profile.get('blog') or user_profile.get('twitter_username'):
        score += 3

    # ── 4. Breadth / activity (10 pts) ───────────────────────────────────────
    score += min(len(repos) / 30, 1.0) * 5
    recent_count = sum(
        1 for r in repos
        if r.get('updated_at') and
        (now - datetime.strptime(r['updated_at'], "%Y-%m-%dT%H:%M:%SZ")
         .replace(tzinfo=timezone.utc)).days <= 365
    )
    score += (recent_count / max(len(repos), 1)) * 5

    return round(min(score, 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_github_profile(github_url, jd_text="", groq_client=None):
    from datetime import datetime, timezone
    username = get_username(github_url)
    repos = fetch_repositories(username)
    user_profile = fetch_user_profile(username)

    if not repos:
        return {
            'github_url': github_url, 'username': username,
            'summary': 'No public repositories found.',
            'top_repositories': [], 'detected_skills': [],
            'total_stars': 0, 'total_repos': 0, 'final_github_score': 0,
        }

    # Own repos only (exclude forks), fallback to all if everything forked
    own_repos = [r for r in repos if not r.get('fork', False)] or repos

    now = datetime.now(timezone.utc)

    def repo_priority(r):
        stars = r.get('stargazers_count', 0)
        updated = r.get('updated_at', '')
        try:
            days_old = (now - datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ")
                        .replace(tzinfo=timezone.utc)).days
            recency = max(0.0, 1.0 - days_old / 730)
        except Exception:
            recency = 0.0
        has_desc = 1 if r.get('description') else 0
        return stars * 0.5 + recency * 10 + has_desc * 2

    repos_sorted = sorted(own_repos, key=repo_priority, reverse=True)
    top_repositories, detected_skills = [], []

    for repo in repos_sorted[:3]:
        repo_name = repo.get('name')
        # Full detection: README (LLM) + dependency files (keyword)
        tech_stack = detect_tech_stack_for_repo(username, repo_name, groq_client)
        detected_skills.extend(tech_stack)
        top_repositories.append({
            'name': repo_name,
            'description': repo.get('description') or 'No description',
            'language': repo.get('language') or 'Unknown',
            'url': repo.get('html_url'),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'tech_stack': tech_stack,
        })

    detected_skills = list(set(detected_skills))
    total_stars = sum(r.get('stargazers_count', 0) for r in repos)
    final_score = calculate_github_score(
        repos, user_profile, detected_skills, jd_text,
        top_repositories=top_repositories
    )

    return {
        'github_url': github_url,
        'username': username,
        'avatar_url': user_profile.get('avatar_url', ''),
        'bio': user_profile.get('bio', ''),
        'followers': user_profile.get('followers', 0),
        'summary': (
            f"@{username} · {len(repos)} repos · {total_stars} stars · "
            f"Skills: {', '.join(detected_skills) or 'N/A'} · Score: {final_score}/100"
        ),
        'top_repositories': top_repositories,
        'detected_skills': detected_skills,
        'total_stars': total_stars,
        'total_repos': len(repos),
        'final_github_score': final_score,
    }
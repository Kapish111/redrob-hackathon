import math
from collections import defaultdict

# =========================
# SKILL ALIASES
# =========================

SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",

    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",

    "typescript": "typescript",
    "typescrpit": "typescript",

    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",

    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",

    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",

    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",

    "feature engineering": "feature_engineering",

    "statistics": "statistics",
    "stats": "statistics",

    "regression": "regression",
    "clustering": "clustering",

    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",

    "pandas": "pandas",
    "numpy": "numpy",

    # Web Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",

    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",

    "redux": "redux",
    "tailwind": "tailwind",

    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",

    "jest": "jest",
    "graphql": "graphql",

    # Web Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",

    "flask": "flask",

    "spring boot": "spring_boot",
    "springboot": "spring_boot",

    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",

    "microservices": "microservices",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",

    "postgresql": "postgresql",
    "postgres": "postgresql",

    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps / Cloud
    "docker": "docker",

    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",

    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",

    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS Fundamentals
    "algorithms": "algorithms",
    "algoritms": "algorithms",

    "data structure": "data_structures",
    "data structures": "data_structures",

    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# =========================
# RESUME DATASET
# =========================

resumes = {
    "Arjun Sharma":
        "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",

    "Priya Nair":
        "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",

    "Rahul Gupta":
        "Java, Spring Boot, MySql, Microservices, Docker, kubernates",

    "Sneha Patel":
        "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",

    "Vikram Singh":
        "C++, Algoritms, Data Structure, competitive programming, python",

    "Ananya Krishnan":
        "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",

    "Karan Mehta":
        "Python, Sklearn, XGboost, feature engineering, SQL, tableau",

    "Deepika Rao":
        "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",

    "Aditya Kumar":
        "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",

    "Meera Iyer":
        "python, R, statistics, ML, regression, clustering, Power-BI"
}

# =========================
# JOB DESCRIPTIONS
# =========================

jds = {
    "JD-1 — Kakao (ML Engineer)":
        "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics",

    "JD-2 — Naver (Backend Engineer)":
        "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis",

    "JD-3 — Line (Frontend Engineer)":
        "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"
}

# =========================
# NORMALIZATION FUNCTION
# =========================

def normalize_skills(skill_string):
    tokens = skill_string.lower().split(",")

    normalized = []

    for token in tokens:
        token = token.strip()

        if token in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[token])

    # Deduplication
    return list(set(normalized))

# =========================
# NORMALIZE RESUMES
# =========================

normalized_resumes = {}

for name, skills in resumes.items():
    normalized_resumes[name] = normalize_skills(skills)

# =========================
# BUILD VOCABULARY
# =========================

vocab_set = set()

for skills in normalized_resumes.values():
    vocab_set.update(skills)

vocabulary = sorted(list(vocab_set))

# =========================
# DOCUMENT FREQUENCY
# =========================

df = defaultdict(int)

for skill in vocabulary:
    for skills in normalized_resumes.values():
        if skill in skills:
            df[skill] += 1

# =========================
# IDF COMPUTATION
# =========================

idf = {}

TOTAL_RESUMES = 10

for skill in vocabulary:
    idf[skill] = math.log(TOTAL_RESUMES / df[skill])

# =========================
# TF-IDF VECTOR
# =========================

def build_resume_vector(skills):

    vector = []

    total_skills = len(skills)

    for word in vocabulary:

        if word in skills:
            tf = 1 / total_skills
            tfidf = tf * idf[word]
            vector.append(tfidf)

        else:
            vector.append(0)

    return vector

resume_vectors = {}

for name, skills in normalized_resumes.items():
    resume_vectors[name] = build_resume_vector(skills)

# =========================
# JD BINARY VECTOR
# =========================

def build_jd_vector(skills):

    normalized = normalize_skills(skills)

    vector = []

    for word in vocabulary:
        if word in normalized:
            vector.append(1)
        else:
            vector.append(0)

    return vector

jd_vectors = {}

for jd_name, skills in jds.items():
    jd_vectors[jd_name] = build_jd_vector(skills)

# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(vec1, vec2):

    dot_product = 0
    norm1 = 0
    norm2 = 0

    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm1 += a * a
        norm2 += b * b

    norm1 = math.sqrt(norm1)
    norm2 = math.sqrt(norm2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot_product / (norm1 * norm2)

# =========================
# RANK CANDIDATES
# =========================

for jd_name, jd_vector in jd_vectors.items():

    results = []

    for candidate, resume_vector in resume_vectors.items():

        score = cosine_similarity(resume_vector, jd_vector)

        results.append((candidate, round(score, 2)))

    # Sort:
    # 1. Descending score
    # 2. Alphabetically for ties

    results.sort(key=lambda x: (-x[1], x[0]))

    top3 = results[:3]

    print("\n" + jd_name)

    formatted = []

    for name, score in top3:
        formatted.append(f"{name}({score})")

    print(", ".join(formatted))

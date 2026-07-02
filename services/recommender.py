import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("vectorstore/shl.index")
with open("vectorstore/catalog.pkl", "rb") as f:
    catalog = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

LABEL_TO_CODE = {
    "Ability & Aptitude": "A",
    "Ability and Aptitude": "A",
    "Biodata & Situational Judgement": "B",
    "Biodata & Situational Judgment": "B",
    "Biodata and Situational Judgement": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Development and 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Knowledge and Skills": "K",
    "Personality & Behavior": "P",
    "Personality and Behavior": "P",
    "Simulations": "S",
}

EXPANSIONS = {
    "java": "java spring hibernate jvm backend developer",
    "python": "python django flask data science scripting backend",
    "javascript": "javascript node react angular frontend web",
    "js": "javascript node react frontend web",
    "react": "react javascript frontend ui components",
    "node": "node.js javascript backend server api",
    "angular": "angular javascript typescript frontend web",
    "spring": "spring boot java microservices backend",
    "aws": "amazon web services cloud ec2 s3 lambda devops",
    "azure": "microsoft azure cloud devops infrastructure",
    "cloud": "aws azure gcp cloud infrastructure devops",
    "docker": "docker containers kubernetes devops microservices",
    "kubernetes": "kubernetes docker containers orchestration devops",
    "devops": "docker kubernetes jenkins ci/cd cloud infrastructure",
    "sql": "sql database queries relational mysql postgresql",
    "data science": "machine learning python statistics data analysis",
    "machine learning": "ml python data science algorithms models",
    "ml": "machine learning data science python algorithms",
    "android": "android kotlin java mobile development",
    "ios": "ios swift mobile apple development",
    "personality": "personality behavior traits occupational opq",
    "behaviour": "personality behavior traits occupational opq",
    "behavior": "personality behavior traits occupational opq",
    "cognitive": "numerical verbal reasoning ability aptitude verify",
    "aptitude": "numerical verbal reasoning ability aptitude verify",
    "reasoning": "numerical verbal inductive deductive ability verify",
    "numerical": "numerical reasoning math data analysis verify",
    "verbal": "verbal reasoning communication english language verify",
    "leadership": "leadership management competencies behavior opq",
    "management": "manager leadership competencies project",
    "sales": "sales customer relationship crm negotiation",
    "customer service": "customer service contact center support situational",
    "finance": "finance accounting financial analysis excel",
    "accounting": "accounts payable receivable financial bookkeeping",
    "hr": "human resources talent management recruitment",
    "project management": "project planning agile scrum stakeholder",
    "agile": "agile scrum kanban software development project",
    "testing": "manual testing automation selenium qa software",
    "qa": "quality assurance testing manual automation selenium",
    "security": "cybersecurity network security information assurance",
    "c#": "csharp dotnet microsoft .net backend asp",
    "dotnet": "c# .net asp.net microsoft backend",
    ".net": "c# dotnet asp.net microsoft backend framework",
    "php": "php web development laravel symfony backend",
    "ruby": "ruby rails web development backend",
    "kotlin": "kotlin android java jvm mobile",
    "swift": "swift ios apple mobile development",
    "typescript": "typescript javascript nodejs angular react",
    "tableau": "tableau data visualization business intelligence",
    "power bi": "power bi data visualization microsoft analytics",
    "salesforce": "salesforce crm sales cloud service cloud",
    "sap": "sap erp enterprise resource planning business",
    "excel": "microsoft excel spreadsheet data analysis",
    "hadoop": "hadoop big data mapreduce hdfs spark",
    "spark": "apache spark big data processing analytics",
    "kafka": "apache kafka streaming data pipeline messaging",
    "scala": "scala functional programming jvm big data spark",
    "go": "golang go programming backend systems",
    "c++": "c++ systems programming performance low level",
    "fullstack": "frontend backend javascript react node sql",
    "full stack": "frontend backend javascript react node sql",
    "full-stack": "frontend backend javascript react node sql",
}


def expand_query(query: str) -> str:
    q = query.lower()
    additions = []
    for keyword, expansion in EXPANSIONS.items():
        if keyword in q:
            additions.append(expansion)
    if additions:
        q = q + " " + " ".join(additions)
    return q


def normalize_test_type(keys_list) -> str:
    if not keys_list:
        return "K"
    if isinstance(keys_list, str):
        keys_list = [keys_list]
    codes = []
    for t in keys_list:
        t = str(t).strip()
        code = LABEL_TO_CODE.get(t)
        if code:
            codes.append(code)
        elif len(t) == 1 and t.isupper():
            codes.append(t)
    seen = set()
    unique_codes = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique_codes.append(c)
    return ",".join(unique_codes) if unique_codes else "K"


def retrieve_assessments(query: str, k: int = 10) -> list:
    expanded = expand_query(query)
    embedding = model.encode([expanded])
    embedding = np.array(embedding).astype("float32")

    fetch_k = min(k * 3, len(catalog))
    distances, indices = index.search(embedding, fetch_k)

    q_lower = query.lower()
    results = []

    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(catalog):
            continue
        item = catalog[idx]
        name = item.get("name", "")
        distance = float(distances[0][i])

        boost = 0
        for word in q_lower.split():
            if len(word) > 2 and word in name.lower():
                boost += 2

        results.append((distance - boost, idx, item))

    results.sort(key=lambda x: x[0])

    recommendations = []
    seen_names = set()

    for _, idx, item in results:
        name = item.get("name", "")
        if name in seen_names:
            continue
        seen_names.add(name)

        url = item.get("link", "")
        test_type = normalize_test_type(item.get("keys", []))
        description = item.get("description", "")[:200]

        recommendations.append({
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": description,
        })

        if len(recommendations) >= k:
            break

    return recommendations
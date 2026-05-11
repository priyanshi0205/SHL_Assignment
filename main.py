import json
import re
from pathlib import Path
from typing import Any, List, Literal, Optional

import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


app = FastAPI()
CATALOG_PATH: Path = Path(__file__).parent / "catalog.json"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
BOOST_KEYWORDS: List[str] = [
    "personality",
    "cognitive",
    "technical",
    "leadership",
    "teamwork",
    "communication",
    "programming",
    "stakeholder",
]
PRODUCT_ALIASES: dict[str, str] = {
    "opq": "Occupational Personality Questionnaire OPQ32r",
    "opq32r": "Occupational Personality Questionnaire OPQ32r",
    "gsa": "Global Skills Assessment",
    "verify g+": "SHL Verify Interactive G+",
    "verify interactive g+": "SHL Verify Interactive G+",
    "general ability test": "SHL Verify Interactive G+",
    "general ability": "SHL Verify Interactive G+",
}

# In-memory objects initialized at startup.
catalog_data: List[dict[str, Any]] = []
embedding_model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.IndexFlatL2] = None


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


@app.on_event("startup")
def startup_event() -> None:
    global catalog_data, embedding_model, faiss_index

    # Load assessment catalog from local JSON file.
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        loaded_catalog: Any = json.load(file)
    catalog_data = loaded_catalog if isinstance(loaded_catalog, list) else []

    # Load sentence-transformers model for semantic embeddings.
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, local_files_only=True
        )
    except Exception:
        embedding_model = None
        faiss_index = None
        return

    # Build searchable text per assessment and create an in-memory FAISS index.
    corpus_texts: List[str] = []
    for assessment in catalog_data:
        tags: List[str] = [str(tag) for tag in assessment.get("tags", [])]
        text_blob: str = " ".join(
            [
                str(assessment.get("name", "")),
                str(assessment.get("description", "")),
                " ".join(tags),
                str(assessment.get("test_type", "")),
            ]
        ).strip()
        corpus_texts.append(text_blob)

    if not corpus_texts:
        faiss_index = None
        return

    try:
        embeddings: np.ndarray = embedding_model.encode(
            corpus_texts, batch_size=8, convert_to_numpy=True
        ).astype("float32")
        dimension: int = int(embeddings.shape[1])
        index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss_index = index
    except Exception:
        embedding_model = None
        faiss_index = None


def detect_role(query: str) -> bool:
    role_keywords: List[str] = [
        "developer",
        "engineer",
        "analyst",
        "manager",
        "architect",
        "tester",
        "consultant",
        "administrator",
        "admin",
        "operator",
        "assistant",
        "agent",
        "staff",
        "trainee",
        "sales",
        "full-stack",
        "full stack",
        "contact center",
        "call center",
    ]
    lowered_query: str = query.lower()
    return any(keyword in lowered_query for keyword in role_keywords)


def detect_experience(query: str) -> bool:
    lowered_query: str = query.lower()
    experience_keywords: List[str] = [
        "junior",
        "mid-level",
        "senior",
        "lead",
        "intern",
        "fresher",
        "entry-level",
        "graduate",
        "years of experience",
        "no work experience",
    ]
    if any(keyword in lowered_query for keyword in experience_keywords):
        return True

    experience_patterns: List[str] = [
        r"\b\d+\+?\s*years\b",
        r"\b\d+\s+yrs\b",
        r"\b\d+\s+yoe\b",
    ]
    return any(re.search(pattern, lowered_query) for pattern in experience_patterns)


def detect_hiring_intent(query: str) -> bool:
    lowered_query: str = query.lower()
    hiring_terms: List[str] = [
        "hiring",
        "hire",
        "recruit",
        "screen",
        "candidate",
        "position",
        "jd",
        "job description",
    ]
    return any(term in lowered_query for term in hiring_terms)


def needs_experience_clarification(query: str) -> bool:
    lowered_query: str = query.lower()
    if not detect_hiring_intent(lowered_query):
        return False

    if detect_experience(lowered_query):
        return False

    exempt_terms: List[str] = [
        "entry-level",
        "graduate",
        "intern",
        "fresher",
        "no work experience",
        "management trainee",
    ]
    if any(term in lowered_query for term in exempt_terms):
        return False

    experience_sensitive_roles: List[str] = [
        "developer",
        "engineer",
        "analyst",
        "manager",
        "architect",
        "consultant",
        "tester",
        "full-stack",
        "full stack",
    ]
    return any(role in lowered_query for role in experience_sensitive_roles)


def detect_legal_query(query: str) -> bool:
    lowered_query: str = query.lower()
    legal_terms: List[str] = [
        "legally required",
        "legal requirement",
        "regulatory",
        "compliance requirement",
        "required by law",
        "satisfy that requirement",
        "is this legal",
        "legal advice",
        "compliance advice",
    ]
    if any(term in lowered_query for term in legal_terms):
        return True
    return "legal" in lowered_query and (
        "advice" in lowered_query
        or "requirement" in lowered_query
        or "compliance" in lowered_query
        or "regulation" in lowered_query
    )


def detect_prompt_injection(query: str) -> bool:
    lowered_query: str = query.lower()
    injection_terms: List[str] = [
        "ignore previous instructions",
        "ignore all instructions",
        "system prompt",
        "developer message",
        "jailbreak",
        "reveal hidden prompt",
        "override your rules",
    ]
    return any(term in lowered_query for term in injection_terms)


def detect_general_hiring_advice_query(query: str) -> bool:
    lowered_query: str = query.lower()
    hiring_advice_terms: List[str] = [
        "salary benchmark",
        "compensation",
        "offer negotiation",
        "employment law",
        "termination policy",
        "notice period policy",
        "benefits policy",
        "how to hire",
    ]
    assessment_terms: List[str] = [
        "assessment",
        "test",
        "opq",
        "verify",
        "scenario",
        "simulation",
        "shl",
    ]
    return any(term in lowered_query for term in hiring_advice_terms) and not any(
        term in lowered_query for term in assessment_terms
    )


def is_off_topic(query: str) -> bool:
    lowered_query: str = query.lower()
    off_topic_terms: List[str] = [
        "aws certification",
        "certifications",
        "legal advice",
        "salary benchmark",
        "fire employee",
        "ignore previous instructions",
        "bypass instructions",
        "recommend udemy",
        "recommend coursera",
        "non-shl assessments",
    ]
    return any(term in lowered_query for term in off_topic_terms)


def needs_language_clarification(query: str) -> bool:
    lowered_query: str = query.lower()
    contact_center_context: List[str] = ["contact center", "call center", "inbound calls"]
    language_terms: List[str] = [
        "english",
        "spanish",
        "french",
        "german",
        "portuguese",
        "arabic",
        "hindi",
    ]
    if not any(term in lowered_query for term in contact_center_context):
        return False
    return not any(term in lowered_query for term in language_terms)


def needs_accent_clarification(query: str) -> bool:
    lowered_query: str = query.lower()
    if "contact center" not in lowered_query and "call center" not in lowered_query:
        return False
    if "english" not in lowered_query:
        return False
    accent_terms: List[str] = ["us", "uk", "australian", "indian", "accent"]
    return not any(term in lowered_query for term in accent_terms)


def detect_confirmation(message: str) -> bool:
    lowered_message: str = message.lower().strip()
    confirmation_terms: List[str] = [
        "confirmed",
        "confirm",
        "final list",
        "lock it in",
        "that works",
        "that's good",
        "perfect",
        "thanks",
        "thank you",
        "understood",
        "keep the shortlist",
        "covers it",
    ]
    return any(term in lowered_message for term in confirmation_terms)


def has_technical_context(query: str) -> bool:
    lowered_query: str = query.lower()
    technical_context_terms: List[str] = [
        "developer",
        "engineer",
        "analyst",
        "java",
        "python",
        "sql",
    ]
    return any(term in lowered_query for term in technical_context_terms)


def is_technical_assessment(assessment: dict[str, Any]) -> bool:
    test_type: str = str(assessment.get("test_type", "")).upper()
    if "K" in test_type:
        return True

    combined_text: str = " ".join(
        [
            str(assessment.get("name", "")).lower(),
            str(assessment.get("description", "")).lower(),
            " ".join(str(tag).lower() for tag in assessment.get("tags", [])),
            " ".join(str(key).lower() for key in assessment.get("keys", [])),
        ]
    )
    technical_markers: List[str] = [
        "technical",
        "knowledge",
        "skills",
        "programming",
        "java",
        "python",
        "sql",
        "spring",
        "docker",
        "aws",
        "coding",
        "software development",
        "backend",
        "frontend",
    ]
    return any(marker in combined_text for marker in technical_markers)


def detect_refinement_focus(query: str) -> set[str]:
    lowered_query: str = query.lower()
    focus: set[str] = set()
    if "personality" in lowered_query or "opq" in lowered_query:
        focus.add("personality")
    if "leadership" in lowered_query or "leader" in lowered_query:
        focus.add("leadership")
    return focus


def is_refinement_match(assessment: dict[str, Any], focus: set[str]) -> bool:
    if not focus:
        return False
    combined_text: str = " ".join(
        [
            str(assessment.get("name", "")).lower(),
            str(assessment.get("description", "")).lower(),
            " ".join(str(tag).lower() for tag in assessment.get("tags", [])),
            str(assessment.get("test_type", "")).lower(),
        ]
    )
    if "personality" in focus and (
        "personality" in combined_text or "opq" in combined_text or " p" in combined_text
    ):
        return True
    if "leadership" in focus and "leadership" in combined_text:
        return True
    return False


def ensure_technical_balance(
    query: str, ranked_results: List[dict[str, Any]], max_items: int = 10
) -> List[dict[str, Any]]:
    if not ranked_results:
        return ranked_results
    focus: set[str] = detect_refinement_focus(query)
    technical_context: bool = has_technical_context(query)
    if not technical_context and not focus:
        return ranked_results[:max_items]

    selected: List[dict[str, Any]] = []

    # Preserve at least one technical assessment when technical role intent exists.
    if technical_context:
        technical_candidate: Optional[dict[str, Any]] = None
        for assessment in ranked_results:
            if is_technical_assessment(assessment):
                technical_candidate = assessment
                break
        if technical_candidate is None:
            query_tokens: set[str] = set(
                re.findall(r"[a-z0-9\+]+", query.lower())
            )
            scored_technical: List[tuple[int, dict[str, Any]]] = []
            for assessment in catalog_data:
                if not is_technical_assessment(assessment):
                    continue
                combined_text: str = " ".join(
                    [
                        str(assessment.get("name", "")).lower(),
                        str(assessment.get("description", "")).lower(),
                        " ".join(str(tag).lower() for tag in assessment.get("tags", [])),
                    ]
                )
                token_hits: int = sum(1 for token in query_tokens if token in combined_text)
                scored_technical.append((token_hits, assessment))
            if scored_technical:
                scored_technical.sort(key=lambda item: item[0], reverse=True)
                technical_candidate = scored_technical[0][1]
        if technical_candidate is not None:
            selected.append(technical_candidate)

    # Add one refinement-aligned assessment (personality/leadership) when requested.
    if focus:
        refinement_candidate: Optional[dict[str, Any]] = None
        for assessment in ranked_results:
            if is_refinement_match(assessment, focus):
                if all(assessment is not existing for existing in selected):
                    refinement_candidate = assessment
                    break
        if refinement_candidate is not None:
            selected.append(refinement_candidate)

    balanced: List[dict[str, Any]] = selected.copy()
    for assessment in ranked_results:
        if any(assessment is existing for existing in balanced):
            continue
        balanced.append(assessment)
        if len(balanced) >= max_items:
            break

    if not balanced:
        return ranked_results[:max_items]
    return balanced


def extract_exclusion_terms(query: str) -> List[str]:
    lowered_query: str = query.lower()
    exclusion_patterns: List[str] = [
        r"\bdrop\s+([a-z0-9\+\-\s]+?)(?=$|[\.,;]|\badd\b|\binclude\b|\bfocus\b)",
        r"\bremove\s+([a-z0-9\+\-\s]+?)(?=$|[\.,;]|\badd\b|\binclude\b|\bfocus\b)",
        r"\bexclude\s+([a-z0-9\+\-\s]+?)(?=$|[\.,;]|\badd\b|\binclude\b|\bfocus\b)",
        r"\bwithout\s+([a-z0-9\+\-\s]+?)(?=$|[\.,;]|\badd\b|\binclude\b|\bfocus\b)",
    ]

    raw_chunks: List[str] = []
    for pattern in exclusion_patterns:
        for match in re.finditer(pattern, lowered_query):
            raw_chunks.append(match.group(1).strip())

    exclusions: List[str] = []
    for chunk in raw_chunks:
        parts: List[str] = [piece.strip() for piece in re.split(r",|/|\band\b", chunk)]
        for part in parts:
            if part and part not in exclusions:
                exclusions.append(part)
    return exclusions


def compute_overlap_count(text: str, keywords: List[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def lexical_rank_assessments(query: str, top_k: int) -> List[dict[str, Any]]:
    lowered_query: str = query.lower()
    query_tokens: set[str] = {
        token
        for token in re.findall(r"[a-z0-9\+]+", lowered_query)
        if len(token) >= 2
    }
    exclusion_terms: List[str] = extract_exclusion_terms(lowered_query)
    matched_keywords: List[str] = [
        keyword for keyword in BOOST_KEYWORDS if keyword in lowered_query
    ]

    scored: List[tuple[float, int]] = []
    for idx, assessment in enumerate(catalog_data):
        tags: List[str] = [str(tag).lower() for tag in assessment.get("tags", [])]
        description_text: str = str(assessment.get("description", "")).lower()
        combined_text: str = " ".join(
            [
                str(assessment.get("name", "")).lower(),
                description_text,
                " ".join(tags),
                str(assessment.get("test_type", "")).lower(),
                " ".join(str(k).lower() for k in assessment.get("keys", [])),
                " ".join(str(level).lower() for level in assessment.get("job_levels", [])),
            ]
        )

        if exclusion_terms and any(term in combined_text for term in exclusion_terms):
            continue

        text_tokens: set[str] = set(re.findall(r"[a-z0-9\+]+", combined_text))
        token_overlap: int = len(query_tokens.intersection(text_tokens))
        keyword_overlap: int = compute_overlap_count(combined_text, matched_keywords)
        if token_overlap == 0 and keyword_overlap == 0:
            continue

        score: float = float(token_overlap) + (1.25 * float(keyword_overlap))
        scored.append((score, idx))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [catalog_data[idx] for _, idx in scored[:top_k]]


def resolve_product_name(term: str) -> str:
    normalized: str = term.lower().strip()
    return PRODUCT_ALIASES.get(normalized, term.strip())


def find_assessment_by_name(name_or_alias: str) -> Optional[dict[str, Any]]:
    target: str = resolve_product_name(name_or_alias).lower()
    for assessment in catalog_data:
        name: str = str(assessment.get("name", "")).strip()
        if name.lower() == target:
            return assessment

    for assessment in catalog_data:
        name = str(assessment.get("name", "")).strip().lower()
        if target and target in name:
            return assessment
    return None


def extract_comparison_terms(query: str) -> Optional[tuple[str, str]]:
    lowered_query: str = query.lower().strip()
    patterns: List[str] = [
        r"difference between (.+?) and (.+)$",
        r"compare (.+?) and (.+)$",
        r"compare (.+?) vs (.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered_query)
        if match:
            left = match.group(1).strip(" ?.,")
            right = match.group(2).strip(" ?.,")
            if left and right:
                return left, right
    return None


def build_comparison_reply(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_name: str = str(left.get("name", "Assessment A"))
    right_name: str = str(right.get("name", "Assessment B"))
    left_type: str = str(left.get("test_type", ""))
    right_type: str = str(right.get("test_type", ""))
    left_duration: str = str(left.get("duration", "") or "Not specified")
    right_duration: str = str(right.get("duration", "") or "Not specified")
    left_keys: List[str] = [str(k) for k in left.get("keys", [])]
    right_keys: List[str] = [str(k) for k in right.get("keys", [])]
    left_desc: str = str(left.get("description", "")).strip()
    right_desc: str = str(right.get("description", "")).strip()
    left_summary: str = left_desc[:180] + ("..." if len(left_desc) > 180 else "")
    right_summary: str = right_desc[:180] + ("..." if len(right_desc) > 180 else "")

    return (
        f"{left_name} vs {right_name}: "
        f"{left_name} is type {left_type} and {right_name} is type {right_type}. "
        f"Duration: {left_duration} vs {right_duration}. "
        f"Key areas: {', '.join(left_keys[:4]) or 'Not specified'} vs "
        f"{', '.join(right_keys[:4]) or 'Not specified'}. "
        f"Catalog descriptions: {left_summary} | {right_summary}"
    )


def compare_assessments(query: str) -> Optional[str]:
    terms: Optional[tuple[str, str]] = extract_comparison_terms(query)
    if terms is None:
        return None

    left_assessment: Optional[dict[str, Any]] = find_assessment_by_name(terms[0])
    right_assessment: Optional[dict[str, Any]] = find_assessment_by_name(terms[1])
    if left_assessment is None or right_assessment is None:
        return None

    left_name: str = str(left_assessment.get("name", "Assessment A"))
    right_name: str = str(right_assessment.get("name", "Assessment B"))
    left_type: str = str(left_assessment.get("test_type", "")).strip() or "Unknown"
    right_type: str = str(right_assessment.get("test_type", "")).strip() or "Unknown"
    left_desc: str = str(left_assessment.get("description", "")).strip()
    right_desc: str = str(right_assessment.get("description", "")).strip()
    left_tags: List[str] = [str(tag).strip().lower() for tag in left_assessment.get("tags", []) if str(tag).strip()]
    right_tags: List[str] = [str(tag).strip().lower() for tag in right_assessment.get("tags", []) if str(tag).strip()]

    left_measures: str = left_desc[:200] + ("..." if len(left_desc) > 200 else "")
    right_measures: str = right_desc[:200] + ("..." if len(right_desc) > 200 else "")

    left_only_tags: List[str] = [tag for tag in left_tags if tag not in right_tags][:3]
    right_only_tags: List[str] = [tag for tag in right_tags if tag not in left_tags][:3]
    shared_tags: List[str] = [tag for tag in left_tags if tag in right_tags][:3]

    left_usage: str = (
        f"typically used when hiring requires {', '.join(left_only_tags)}"
        if left_only_tags
        else f"typically used for roles aligned to test type {left_type}"
    )
    right_usage: str = (
        f"typically used when hiring requires {', '.join(right_only_tags)}"
        if right_only_tags
        else f"typically used for roles aligned to test type {right_type}"
    )
    shared_text: str = (
        f"Both also touch on {', '.join(shared_tags)}."
        if shared_tags
        else ""
    )

    return (
        f"{left_name} focuses on {left_measures} "
        f"while {right_name} focuses on {right_measures} "
        f"Key difference: {left_name} is test type {left_type} and {right_name} is test type {right_type}. "
        f"{left_name} is {left_usage}, and {right_name} is {right_usage}. "
        f"{shared_text}".strip()
    )


def search_assessments(query: str, top_k: int = 5) -> List[dict[str, Any]]:
    if not query.strip():
        return []
    if not catalog_data:
        return []

    bounded_top_k: int = max(1, min(top_k, 10))

    if embedding_model is None or faiss_index is None:
        return lexical_rank_assessments(query, bounded_top_k)

    # Embed user query and retrieve nearest neighbors from the FAISS index.
    query_vector: np.ndarray = embedding_model.encode(
        [query], convert_to_numpy=True
    ).astype("float32")

    candidate_limit: int = min(max(bounded_top_k * 4, 10), len(catalog_data))
    distances, indices = faiss_index.search(query_vector, candidate_limit)

    lowered_query: str = query.lower()
    matched_keywords: List[str] = [
        keyword for keyword in BOOST_KEYWORDS if keyword in lowered_query
    ]
    exclusion_terms: List[str] = extract_exclusion_terms(lowered_query)

    # Re-rank FAISS candidates with deterministic keyword overlap boosting.
    scored_candidates: List[tuple[float, int]] = []
    for rank_position, raw_idx in enumerate(indices[0]):
        idx: int = int(raw_idx)
        if idx < 0 or idx >= len(catalog_data):
            continue

        assessment: dict[str, Any] = catalog_data[idx]
        tags: List[str] = [str(tag).lower() for tag in assessment.get("tags", [])]
        description_text: str = str(assessment.get("description", "")).lower()
        combined_text: str = " ".join(
            [
                str(assessment.get("name", "")).lower(),
                description_text,
                " ".join(tags),
                str(assessment.get("test_type", "")).lower(),
            ]
        )

        if exclusion_terms and any(term in combined_text for term in exclusion_terms):
            continue

        tag_overlap: int = sum(1 for keyword in matched_keywords if keyword in tags)
        description_overlap: int = compute_overlap_count(description_text, matched_keywords)
        total_overlap: int = compute_overlap_count(combined_text, matched_keywords)

        semantic_score: float = -float(distances[0][rank_position])
        boost_score: float = (
            (0.85 * tag_overlap) + (0.50 * description_overlap) + (0.25 * total_overlap)
        )
        final_score: float = semantic_score + boost_score
        scored_candidates.append((final_score, idx))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))

    unique_indices: List[int] = []
    for _, idx in scored_candidates:
        if idx not in unique_indices:
            unique_indices.append(idx)
        if len(unique_indices) >= bounded_top_k:
            break

    return [catalog_data[idx] for idx in unique_indices]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Combine all user turns into one retrieval query.
    user_messages: List[str] = [
        message.content.strip()
        for message in request.messages
        if message.role == "user" and message.content.strip()
    ]
    query: str = " ".join(user_messages).strip()
    latest_user_message: str = user_messages[-1] if user_messages else ""

    if is_off_topic(latest_user_message):
        return ChatResponse(
            reply="I can only help with SHL assessment recommendations.",
            recommendations=[],
            end_of_conversation=False,
        )

    if detect_prompt_injection(latest_user_message):
        return ChatResponse(
            reply="I can only help with SHL assessment selection and cannot follow requests to override system instructions.",
            recommendations=[],
            end_of_conversation=False,
        )

    if detect_general_hiring_advice_query(latest_user_message):
        return ChatResponse(
            reply="I can help with SHL assessment recommendations, comparisons, and shortlist refinement only.",
            recommendations=[],
            end_of_conversation=False,
        )

    if detect_legal_query(latest_user_message):
        return ChatResponse(
            reply=(
                "I can help with assessment selection, but I can't provide legal or "
                "regulatory advice. Please check with your legal or compliance team."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    comparison_reply: Optional[str] = compare_assessments(latest_user_message)
    if comparison_reply is not None:
        return ChatResponse(
            reply=comparison_reply,
            recommendations=[],
            end_of_conversation=False,
        )

    comparison_terms: Optional[tuple[str, str]] = extract_comparison_terms(
        latest_user_message
    )
    if comparison_terms is not None:
        return ChatResponse(
            reply="I couldn't find one or both assessments in the SHL catalog. Please share the exact assessment names.",
            recommendations=[],
            end_of_conversation=False,
        )

    if not detect_role(query):
        return ChatResponse(
            reply="What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    if needs_language_clarification(query):
        return ChatResponse(
            reply="What language are the calls in?",
            recommendations=[],
            end_of_conversation=False,
        )

    if needs_accent_clarification(query):
        return ChatResponse(
            reply="Which English accent should the spoken-language screen target (US, UK, Australian, or Indian)?",
            recommendations=[],
            end_of_conversation=False,
        )

    if needs_experience_clarification(query):
        return ChatResponse(
            reply="What experience level are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    matched_assessments: List[dict[str, Any]] = search_assessments(query, top_k=10)

    if not matched_assessments:
        return ChatResponse(
            reply="I couldn't find a close match in the current SHL catalog data. Could you refine the role or skills?",
            recommendations=[],
            end_of_conversation=False,
        )

    bounded_results: List[dict[str, Any]] = ensure_technical_balance(
        query, matched_assessments, max_items=10
    )[:10]
    recommendation_items: List[Recommendation] = [
        Recommendation(
            name=str(assessment.get("name", "")),
            url=str(assessment.get("url", "")),
            test_type=str(assessment.get("test_type", "")),
        )
        for assessment in bounded_results
    ]

    has_ready_context: bool = detect_role(query) and not needs_language_clarification(
        query
    ) and not needs_accent_clarification(query) and not needs_experience_clarification(
        query
    )
    end_conversation: bool = bool(recommendation_items) and has_ready_context

    return ChatResponse(
        reply="Here are the top matching SHL assessments from the catalog.",
        recommendations=recommendation_items,
        end_of_conversation=end_conversation,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

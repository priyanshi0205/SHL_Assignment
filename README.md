# SHL Conversational Assessment Recommender

This project implements a stateless FastAPI service for conversational SHL assessment recommendations.

## What It Does

- Clarifies vague hiring requests before recommending.
- Recommends 1 to 10 SHL assessments from the local catalog only.
- Supports refinement (for example: add personality tests, include leadership).
- Supports grounded comparison (for example: OPQ vs G+), using catalog data.
- Refuses off-topic, legal/compliance advice, and prompt-injection requests.

## Tech Stack

- Python
- FastAPI
- Pydantic
- Uvicorn
- sentence-transformers
- FAISS

## Project Files

- `main.py` - FastAPI app and chat logic
- `catalog.json` - normalized SHL assessment catalog used by retrieval
- `eval.py` - local evaluation runner (hard checks, probes, Recall@10 approximation)
- `evaluation_results.json` - latest evaluation output
- `evaluation_summary.md` - human-readable evaluation summary
- `Approach_Document_SHL.docx` - submission approach document
- `SUBMISSION_CHECKLIST.md` - submission checklist

## Setup

1. Create/activate virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run API

Service available on : 

## API Endpoints

### GET `/health`

Response:

```json
{"status":"ok"}
```

### POST `/chat`

Request:

```json
{
  "messages": [
    {"role":"user","content":"Hiring a Java developer"},
    {"role":"assistant","content":"What experience level are you hiring for?"},
    {"role":"user","content":"Mid-level around 4 years"}
  ]
}
```

Response schema:

```json
{
  "reply": "string",
  "recommendations": [
    {"name":"string","url":"https://www.shl.com/...","test_type":"string"}
  ],
  "end_of_conversation": false
}
```

Notes:

- `recommendations` is empty during clarification/refusal/comparison turns.
- `recommendations` is 1 to 10 items when shortlist is returned.
- API is stateless: pass full conversation history every call.

## Run Local Evaluation

```powershell
python eval.py --mode local
```

This generates:

- `evaluation_results.json`
- `evaluation_summary.md`

## Evaluate Deployed Endpoint

```powershell
python eval.py --mode http --base-url https://your-api-url
```


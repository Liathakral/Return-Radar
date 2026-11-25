from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from radar.classify import classify
from radar.discover import discover
from radar.llm import get_client
from radar.rank import top_k
from radar.themes import THEMES

app = FastAPI(title="Return Radar API")
client = get_client()  # None when OPENAI_API_KEY is not set


class AnalyzeIn(BaseModel):
    comments: list[str] = Field(min_length=1, max_length=500)
    top_k: int = Field(3, ge=1, le=len(THEMES))


@app.get("/health")
def health():
    return {"ok": True, "llm_configured": client is not None}


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    if client is None:
        raise HTTPException(503, "Set OPENAI_API_KEY to enable classification.")
    comments = [c.strip() for c in body.comments if c.strip()]
    if not comments:
        raise HTTPException(422, "No non-empty comments.")

    labels = classify(comments, client)

    counts = {t: 0 for t in THEMES}
    examples = {t: [] for t in THEMES}
    unmatched, failed = [], 0
    for comment, found in zip(comments, labels):
        if found is None:
            failed += 1
            continue
        if not found:
            unmatched.append(comment)
        for t in found:
            counts[t] += 1
            if len(examples[t]) < 2:
                examples[t].append(comment)

    answered = len(comments) - failed
    top = [
        {
            "theme": THEMES[t]["name"],
            "count": n,
            "share": round(n / answered, 3) if answered else 0,
            "examples": examples[t],
            "fix": THEMES[t]["fix"],
        }
        for t, n in top_k(counts, body.top_k)
    ]
    return {
        "total": len(comments),
        "failed": failed,
        "top": top,
        "emerging": discover(unmatched, client),
    }

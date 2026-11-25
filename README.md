# Return Radar API

Reads customer return and review comments, finds why customers were unhappy, and ranks the product-page fixes that would prevent the most returns.

Independent concept project. Not affiliated with Amazon. Sample data is synthetic.

## How a comment becomes a ranked fix

```
comments ──► classify (LLM, structured output) ──► count per cause ──► top-K (min-heap) ──► ranked fixes
                         │
                         └─ no cause matched ──► embed ──► k-means ──► LLM names each group ──► "emerging issues"
```

1. **Classify.** Comments go to the LLM in batches of 20. The model must answer in a JSON schema where every cause is one of our ids (`radar/themes.py`), so there is no free-text parsing. A comment can have several causes.
2. **Count and rank.** Causes are counted in a dict, and a size-K min-heap keeps the K largest (`radar/rank.py`).
3. **Discover.** Comments that matched nothing are embedded and clustered with k-means. The LLM names each large cluster from its three most central comments. This surfaces problems the fixed list does not know about.

## Why these choices

- **LLM for labels, embeddings for discovery.** Labels need judgment ("battery lasts 3 hours, listing says 8" is a performance gap, not a delivery issue), which an LLM does well. Embeddings plus clustering find new groups without a predefined list, but cannot name a cause on their own.
- **Taxonomy in plain language.** Adding a cause is one new entry in `themes.py`. No retraining.
- **Failures are counted.** If a batch fails, those comments are returned as `failed` instead of being treated as "no cause", so a rate-limit never quietly changes the ranking.
- **Comments are untrusted input.** The prompt tells the model to treat them as data, output is schema-constrained, and any unknown cause or out-of-range index is dropped. This limits prompt injection but does not remove the risk.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                 # 7 offline tests, no API key needed
export OPENAI_API_KEY=...           # optional model overrides: RADAR_LLM_MODEL, RADAR_EMBED_MODEL
uvicorn app:app --reload
```

```bash
curl -s localhost:8000/analyze -H 'content-type: application/json' -d '{
  "top_k": 3,
  "comments": ["Straps are way too short", "Zipper broke after two weeks", "Runs small", "App keeps crashing"]
}'
```


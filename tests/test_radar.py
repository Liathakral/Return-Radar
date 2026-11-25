"""Offline tests. A fake client stands in for OpenAI, so no key or network is needed.
They check OUR logic (batching, validation, ranking, endpoint), not the model's accuracy."""
import json
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import HashingVectorizer

import app as app_module
from radar.classify import classify
from radar.discover import discover
from radar.rank import top_k

RULES = {"fit": ["too short", "too small", "runs small"], "dur": ["broke", "snapped"],
         "dmg": ["crushed", "torn"], "ship": ["delivery took"]}


class FakeClient:
    """Mimics the two OpenAI calls we use."""

    def __init__(self, bad_batches=0, junk=False):
        self.bad_batches, self.junk = bad_batches, junk
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _chat(self, **kw):
        if "response_format" not in kw:  # cluster-naming call
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Fake name"))])
        if self.bad_batches > 0:
            self.bad_batches -= 1
            raise RuntimeError("api down")
        items = json.loads(kw["messages"][1]["content"])
        rows = [{"i": it["i"], "themes": [t for t, ws in RULES.items() if any(w in it["text"].lower() for w in ws)]}
                for it in items]
        if self.junk:  # out-of-range index and unknown theme must be ignored
            rows += [{"i": 999, "themes": ["fit"]}, {"i": 0, "themes": ["not_a_theme"]}]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"results": rows})))])

    def _embed(self, model, input):
        X = HashingVectorizer(n_features=128, alternate_sign=False).transform(input).toarray()
        return SimpleNamespace(data=[SimpleNamespace(embedding=list(r)) for r in X])


def test_classify_labels_and_batches():
    comments = ["Straps too short", "Zipper broke", "Love it"] * 15  # 45 comments -> 3 batches
    labels = classify(comments, FakeClient(), batch_size=20)
    assert len(labels) == 45
    assert labels[0] == ["fit"] and labels[1] == ["dur"] and labels[2] == []


def test_classify_ignores_junk_from_model():
    labels = classify(["Straps too short", "Love it"], FakeClient(junk=True))
    assert labels == [["fit"], []]  # unknown theme dropped, out-of-range row ignored, valid label kept


def test_failed_batch_is_reported_not_hidden():
    labels = classify(["a"] * 25, FakeClient(bad_batches=1), batch_size=20)
    assert labels[:20] == [None] * 20
    assert labels[20:] == [[]] * 5


def test_top_k_heap_matches_sorting_and_breaks_ties_by_order():
    counts = {"a": 3, "b": 7, "c": 0, "d": 7, "e": 5, "f": 1}
    assert top_k(counts, 3) == [("b", 7), ("d", 7), ("e", 5)]
    assert top_k(counts, 10) == sorted([(k, v) for k, v in counts.items() if v], key=lambda x: (-x[1], list(counts).index(x[0])))


def test_discover_groups_unmatched_comments():
    comments = ["app crashes on login", "login screen crashes the app", "crashes when I log in",
                "colour is ugly pink", "ugly pink colour", "the pink colour looks ugly"]
    groups = discover(comments, FakeClient(), k=2)
    assert sorted(g["size"] for g in groups) == [3, 3]
    assert all(g["label"] == "Fake name" for g in groups)


def test_endpoint_end_to_end():
    app_module.client = FakeClient()
    http = TestClient(app_module.app)
    comments = ["Straps too short", "runs small", "Zipper broke", "Box crushed", "Love it"]
    r = http.post("/analyze", json={"comments": comments, "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5 and body["failed"] == 0
    assert body["top"][0]["theme"] == "Size and fit" and body["top"][0]["count"] == 2
    assert len(body["top"]) == 2


def test_endpoint_without_key_is_503():
    app_module.client = None
    assert TestClient(app_module.app).post("/analyze", json={"comments": ["x"]}).status_code == 503

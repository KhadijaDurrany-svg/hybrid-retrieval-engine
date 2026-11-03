from fastapi.testclient import TestClient

from hre.service import create_app


def _client():
    app = create_app(data_dir="/nonexistent-on-purpose")
    return TestClient(app)


def test_healthz_before_index():
    with _client() as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["indexed"] is False


def test_search_without_index_409():
    with _client() as c:
        r = c.post("/search", json={"query": "vector"})
        assert r.status_code == 409


def test_index_then_search():
    with _client() as c:
        docs = {
            "documents": [
                {"id": "a", "text": "vector embedding nearest neighbour search", "title": "ann"},
                {"id": "b", "text": "bm25 inverted index term frequency", "title": "lexical"},
                {"id": "c", "text": "retrieval augmented generation grounding", "title": "rag"},
            ]
        }
        r = c.post("/index", json=docs)
        assert r.status_code == 200
        assert r.json()["n_docs"] == 3

        r = c.post("/search", json={"query": "vector nearest neighbour", "k": 2, "mode": "hybrid"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["hits"]) <= 2
        assert "timings" in body


def test_index_rejects_duplicate_ids():
    with _client() as c:
        docs = {"documents": [{"id": "x", "text": "a b"}, {"id": "x", "text": "c d"}]}
        r = c.post("/index", json=docs)
        assert r.status_code == 400

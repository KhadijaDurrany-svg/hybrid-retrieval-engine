from hre.analysis import Analyzer


def test_lowercase_and_accents():
    a = Analyzer()
    assert a.analyze("Café RÉSUMÉ") == a.analyze("cafe resume")


def test_stopwords_removed():
    a = Analyzer(stem=False)
    assert "the" not in a.analyze("the quick brown fox")
    assert a.analyze("the quick brown fox") == ["quick", "brown", "fox"]


def test_stopwords_kept_when_disabled():
    a = Analyzer(remove_stopwords=False, stem=False)
    assert "the" in a.analyze("the fox")


def test_mild_stemmer_plural():
    a = Analyzer()
    # "vectors" -> "vector", "embeddings" -> "embedding"
    assert a.analyze("vectors") == a.analyze("vector")


def test_empty_and_none():
    a = Analyzer()
    assert a.analyze("") == []
    assert a.analyze(None) == []


def test_numbers_tokenized():
    a = Analyzer()
    assert "bm25" in a.analyze("BM25 ranking")

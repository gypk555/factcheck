import math

import numpy as np
import pytest

from truthlens.model import Predictor, ngrams, tokenize


@pytest.fixture(scope="module")
def predictor():
    return Predictor.load()


def tiny_model():
    terms = ["shocking", "said", "on thursday", "the"]
    return Predictor(terms, idf=np.array([2.0, 1.5, 3.0, 1.0]),
                     coef=np.array([3.0, -2.0, -1.0, 0.0]), bias=0.1)


def test_tokenize_matches_sklearn_rule():
    # Words of 2+ letters/digits, lower-cased; single characters are dropped.
    assert tokenize("Hello, World 42 a") == ["hello", "world", "42"]
    assert tokenize("It's O.K.") == ["it"]


def test_ngrams():
    assert ngrams(["a1", "b2", "c3"], 2) == ["a1", "b2", "c3", "a1 b2", "b2 c3"]


def test_tfidf_by_hand():
    m = tiny_model()
    vec = m.vectorize("shocking shocking said")
    raw = {0: (1 + math.log(2)) * 2.0, 1: 1.0 * 1.5}
    norm = math.sqrt(sum(v * v for v in raw.values()))
    assert vec == pytest.approx({j: v / norm for j, v in raw.items()})


def test_contributions_sum_to_score():
    m = tiny_model()
    p = m.predict("shocking news said on thursday")
    weights = sum(t["weight"] for t in p.top_fake_terms + p.top_real_terms)
    expected = 1 / (1 + math.exp(-(m.bias + weights)))
    assert p.prob_fake == pytest.approx(expected, abs=1e-3)


def test_no_known_words_gives_bias_only():
    m = tiny_model()
    p = m.predict("zzz qqq www")
    assert p.prob_fake == pytest.approx(1 / (1 + math.exp(-0.1)), abs=1e-4)
    assert p.known_word_ratio == 0.0


def test_highlights_rebuild_cleaned_text():
    m = tiny_model()
    p = m.predict("Shocking!  Officials said on Thursday.")
    assert "".join(s["text"] for s in p.highlights) == "Shocking! Officials said on Thursday."


def test_inconsistent_files_rejected():
    with pytest.raises(ValueError):
        Predictor(["a"], np.array([1.0, 2.0]), np.array([1.0]), 0.0)


# ----- trained model behaviour
def test_trained_model_loads_without_pickle(predictor):
    assert len(predictor.terms) > 10_000
    assert predictor.meta["test_metrics"]["accuracy"] > 0.9


def test_sensational_text_is_fake(predictor):
    p = predictor.predict("BREAKING: You won't believe this SHOCKING video! Watch before it gets deleted, share now!")
    assert p.label == "FAKE"


def test_wire_style_text_is_real(predictor):
    p = predictor.predict(
        "The council voted on Thursday to approve the budget, officials said. The plan increases "
        "spending on buses by 4 percent, the committee chair said in a statement."
    )
    assert p.label == "REAL"


def test_short_text_warns(predictor):
    assert any("Short text" in w for w in predictor.predict("Hot water kills the flu").warnings)

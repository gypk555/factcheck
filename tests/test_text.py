from truthlens.text import clean_text, join_title_body


def test_removes_reuters_dateline():
    assert clean_text("WASHINGTON (Reuters) - The Senate voted on Tuesday.") == "The Senate voted on Tuesday."


def test_removes_outlet_names_and_credits():
    assert "breitbart" not in clean_text("Big news - Breitbart").lower()
    assert "new york times" not in clean_text("Vote case - The New York Times").lower()
    assert clean_text("Featured image via Getty Images") == ""


def test_removes_urls_and_handles():
    out = clean_text("Look @someone pic.twitter.com/abc https://t.co/x www.site.com end")
    assert out == "Look end"


def test_keeps_normal_words_and_paragraphs():
    assert clean_text("Born via Medicaid") == "Born via Medicaid"
    assert clean_text("Title\n\n\n\nBody   text") == "Title\n\nBody text"


def test_normalises_quotes():
    assert clean_text("“Hi” it’s") == "\"Hi\" it's"


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_join_title_body():
    assert join_title_body("Head", "Body") == "Head . Body"
    assert join_title_body("", "Body") == "Body"
    assert join_title_body(None, None) == ""

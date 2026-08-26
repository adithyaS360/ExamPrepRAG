from app.chunking import chunk_page


def test_module_heading_and_page_are_preserved():
    chunks = chunk_page("syllabus.pdf", 7, "Module 3: Normalization. Functional dependencies.", "abc")
    assert chunks[0].page == 7
    assert chunks[0].heading == "Module 3: Normalization. Functional dependencies."

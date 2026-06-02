import generators.llm_generator as llm_generator

class DummyGemini:
    def __init__(self, *args, **kwargs):
        pass

def test_prompt_contains_grounding_rules(monkeypatch):
    monkeypatch.setattr(llm_generator, "Gemini", DummyGemini)
    gen = llm_generator.ResponseGenerator()
    prompt = gen.qa_prompt_tmpl.template
    assert "không được suy luận" in prompt.lower()
    assert "[file:" in prompt.lower()
    # The prompt must include the evidence_status placeholder used during formatting
    assert "{evidence_status}" in prompt

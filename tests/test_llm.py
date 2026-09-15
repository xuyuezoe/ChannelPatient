import pytest
from scc.llm import ChatClient


@pytest.mark.api
def test_chat_and_cache(tmp_path):
    c = ChatClient("qwen3.7-plus", cache_dir=tmp_path)
    t = c.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=10, temperature=0)
    assert "ok" in t.lower() and c.calls == 1
    t2 = c.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=10, temperature=0)
    assert t2 == t and c.calls == 1 and c.cache_hits == 1
    d = c.chat([{"role": "user", "content": "Give a=1 and b='x'."}], max_tokens=40, temperature=0,
               json_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": False})
    assert d["a"] == 1 and c.calls == 2


def test_parse_json_variants():
    p = ChatClient._parse_json
    assert p('{"a": 1}') == {"a": 1}
    assert p('```json\n{"a": 1}\n```') == {"a": 1}
    assert p('<think>hmm</think>\nSure: {"a": [1,2]} done') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        p("no json here")

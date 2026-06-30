from __future__ import annotations

from pathlib import Path


def test_dashboard_templates_escape_html_like_dynamic_values() -> None:
    app_js = Path("console/app.js").read_text(encoding="utf-8")
    html_like_fixture = {
        "stream_id": "<cam-01>",
        "class_name": "<img src=x onerror=alert(1)>",
        "false_negative_class": "<script>alert(1)</script>",
        "policy_name": "<policy>",
        "detector": "<detector>",
        "verifier": "<verifier>",
        "stream_name": "<Lobby>",
        "location": "<Floor 1>",
    }

    assert html_like_fixture
    required_escapes = [
        "${esc(a.stream_id)}",
        "${esc(a.class_name)}",
        "${esc(a.false_negative_class)}",
        "${esc(p.name)}",
        "${esc(d)}",
        "${esc(p.verifier)}",
        "${esc(s.name)}",
        "${esc(s.location)}",
        "${esc(o.t)}",
        "${esc(o.v)}",
    ]
    for snippet in required_escapes:
        assert snippet in app_js

    unsafe_insertions = [
        "${a.stream_id}</span>",
        "${a.class_name}</div>",
        "${a.false_negative_class}</b>",
        "${p.name}</h3>",
        "${d}</span>",
        '"${p.verifier}"',
        "${s.name}</span>",
        "${s.location}</div>",
        ">${o.t}</option>",
        'value="${o.v}"',
    ]
    for snippet in unsafe_insertions:
        assert snippet not in app_js

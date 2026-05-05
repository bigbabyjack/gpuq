import json

import pytest

from gpuq import protocol as p


def test_submit_request_roundtrip():
    req = p.Submit(
        cmd=["python", "x.py"], cwd="/tmp", env={"A": "1"}, tag="train", priority=5, detach=False
    )
    line = p.encode_request(req)
    assert json.loads(line)["op"] == "submit"
    assert p.decode_request(line) == req


def test_unknown_op_rejected():
    bad = json.dumps({"op": "delete_universe"}) + "\n"
    with pytest.raises(p.ProtocolError):
        p.decode_request(bad)


def test_state_event_roundtrip():
    ev = p.StateEvent(id="j_abc", state="running", pid=42)
    assert p.decode_event(p.encode_event(ev)) == ev


def test_log_event():
    ev = p.LogEvent(id="j_abc", stream="stdout", line="hi")
    assert p.decode_event(p.encode_event(ev)) == ev


def test_each_line_terminated_by_newline():
    ev = p.StateEvent(id="j_x", state="queued", position=1)
    assert p.encode_event(ev).endswith("\n")


def test_decode_rejects_malformed_json():
    with pytest.raises(p.ProtocolError):
        p.decode_request("{not json\n")


def test_attach_from_keyword_translated():
    req = p.Attach(id="j_a", follow=True, from_="start")
    line = p.encode_request(req)
    assert json.loads(line)["from"] == "start"
    assert p.decode_request(line) == req


def test_submit_next_flag_roundtrip():
    req = p.Submit(cmd=["x"], cwd="/", next_=True)
    line = p.encode_request(req)
    assert json.loads(line)["next"] is True
    assert p.decode_request(line) == req


def test_submit_default_next_is_false():
    req = p.Submit(cmd=["x"], cwd="/")
    assert req.next_ is False
    assert json.loads(p.encode_request(req))["next"] is False


def test_bump_request_roundtrip():
    req = p.Bump(id="j_abc")
    line = p.encode_request(req)
    assert json.loads(line)["op"] == "bump"
    assert p.decode_request(line) == req


def test_result_event_roundtrip():
    ev = p.ResultEvent(payload={"jobs": [{"id": "j_x"}]})
    assert p.decode_event(p.encode_event(ev)) == ev

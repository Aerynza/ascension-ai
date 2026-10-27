from src.serving.request_integrity import payload_sha256


def test_key_order_is_not_significant():
    assert payload_sha256({"a": 1, "b": 2}) == payload_sha256({"b": 2, "a": 1})


def test_mutated_payload_changes_digest():
    assert payload_sha256({"value": "original"}) != payload_sha256({"value": "modified"})

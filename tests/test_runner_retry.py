from safety_one.eval.runner import _retryable


def test_transient_errors_retry_but_content_filter_is_final():
    assert _retryable({"error": "HTTPError: HTTP Error 429: Too Many Requests"})
    assert _retryable({"error": "ValueError: provider response did not contain JSON"})
    assert not _retryable({"error": "FrontierContentFilter: x returned finish_reason=content_filter"})
    assert not _retryable({"error": None})

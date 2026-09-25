import pytest

from anomaly.streaming.redis_source import RedisAlertSink, RedisStreamSource


def test_redis_source_raises_clear_error_without_redis_installed():
    """This repo's test suite deliberately does NOT install the `redis`
    package (it's optional -- see requirements.txt / AGENTS.md section 3).
    Constructing RedisStreamSource without it must fail with a clear,
    actionable message, not an opaque NameError/AttributeError deep inside
    the class.
    """
    with pytest.raises(ImportError, match="redis"):
        RedisStreamSource()


def test_redis_alert_sink_raises_clear_error_without_redis_installed():
    with pytest.raises(ImportError, match="redis"):
        RedisAlertSink()

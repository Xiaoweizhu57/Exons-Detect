import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from exons_detect import ExonsDetect, Exons_Detect


def test_public_api_exports_backward_compatible_alias():
    assert ExonsDetect is Exons_Detect

import exons_detect


def test_public_module_exports_names():
    assert "ExonsDetect" in exons_detect.__all__
    assert "Exons_Detect" in exons_detect.__all__

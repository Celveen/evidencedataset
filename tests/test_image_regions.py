from evidencetree.utils.image_regions import safe_crop_box


def test_safe_crop_box_accepts_normalized_region():
    assert safe_crop_box((100, 50), (0.1, 0.2, 0.9, 0.8)) == (10, 10, 90, 40)


def test_safe_crop_box_accepts_pixel_region_and_clamps():
    assert safe_crop_box((100, 50), (-10, 5, 100000, 100000)) == (0, 5, 100, 50)


def test_safe_crop_box_rejects_bad_region():
    assert safe_crop_box((100, 50), (0.5, 0.5, 0.5, 0.8)) is None
    assert safe_crop_box((100, 50), (float("nan"), 0.0, 1.0, 1.0)) is None

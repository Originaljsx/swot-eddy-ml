import pytest
from swot_eddy.pet_version import set_pet_version_in_text

def test_replaces_null():
    text = "pet:\n  pet_version: null\n  params: {}\n"
    out = set_pet_version_in_text(text, "3.6.1")
    assert 'pet_version: "3.6.1"' in out
    assert "null" not in out.split("\n")[1]

def test_replaces_existing_value():
    text = 'pet:\n  pet_version: "0.0.0"\n'
    out = set_pet_version_in_text(text, "3.6.1")
    assert 'pet_version: "3.6.1"' in out
    assert "0.0.0" not in out

def test_preserves_indentation():
    text = "pet:\n  pet_version: null\n"
    out = set_pet_version_in_text(text, "3.6.1")
    assert out.startswith("pet:\n  pet_version:")

def test_missing_line_raises():
    with pytest.raises(ValueError):
        set_pet_version_in_text("pet: {}\n", "3.6.1")

import textwrap
import pytest
from swot_eddy import config as cfg_mod

SAMPLE = textwrap.dedent('''
paths:
  windows:
    data_root: "C:/data"
    out_root: "C:/out"
  colab:
    data_root: "/content/drive/MyDrive/data"
    out_root: "/content/drive/MyDrive/out"
region: {lat_min: 23, lat_max: 42, lon_min: -82, lon_max: -69}
product:
  product_version: "3.0"
  variables:
    "3.0": {ssha: ssha_filtered, mdt: mdt, ugos: ugos_filtered, vgos: vgos_filtered, quality_flag: quality_flag}
tiling: {tile_size: 128, stride: 64, pad_to: 128}
pet: {pet_version: null, params: {shape_error: 85}}
''')

@pytest.fixture
def sample_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return p

def test_windows_paths(sample_config):
    c = cfg_mod.load_config(sample_config, platform="windows")
    assert c["data_root"] == "C:/data"
    assert c["out_root"] == "C:/out"
    assert c["platform"] == "windows"

def test_colab_paths(sample_config):
    c = cfg_mod.load_config(sample_config, platform="colab")
    assert c["data_root"] == "/content/drive/MyDrive/data"

def test_region_box(sample_config):
    c = cfg_mod.load_config(sample_config, platform="windows")
    assert (c["region"]["lat_min"], c["region"]["lat_max"]) == (23, 42)
    assert (c["region"]["lon_min"], c["region"]["lon_max"]) == (-82, -69)

def test_variable_map(sample_config):
    c = cfg_mod.load_config(sample_config, platform="windows")
    vm = cfg_mod.variable_map(c)
    assert vm["ssha"] == "ssha_filtered"
    assert vm["ugos"] == "ugos_filtered"

def test_unknown_platform_raises(sample_config):
    with pytest.raises(KeyError):
        cfg_mod.load_config(sample_config, platform="mars")

def test_env_override(monkeypatch):
    monkeypatch.setenv("SWOT_EDDY_ENV", "colab")
    assert cfg_mod.detect_platform() == "colab"

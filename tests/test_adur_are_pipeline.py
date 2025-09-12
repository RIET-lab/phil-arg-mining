import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'moralkg' is importable in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Provide a tiny stub for 'rootutils' so importing moralkg.config in tests does not fail.
import types
if "rootutils" not in sys.modules:
    stub = types.ModuleType("rootutils")
    def _setup_root(path, indicator=None):
        # Return repository root path string
        return str(ROOT)
    stub.setup_root = _setup_root
    sys.modules["rootutils"] = stub

import pytest

from moralkg.snowball.phase_1.models import registry
from moralkg.snowball.phase_1.models.adapters import normalize_adur_output, normalize_are_output


def test_normalize_adur_output_happy_path():
    raw = {
        "adus": [
            {"id": "a1", "text": "The model performs well.", "label": "claim", "start": 0, "end": 23, "score": 0.98},
            {"id": "a2", "text": "Because it was trained on more data.", "label": "premise", "start": 24, "end": 58, "score": 0.95},
        ]
    }
    out = normalize_adur_output(raw, source_text="The model performs well. Because it was trained on more data.")
    assert "adus" in out and "statistics" in out
    assert out["statistics"]["total_adus"] == 2
    types = out["statistics"]["adu_types"]
    assert types.get("Claim", 0) == 2 or sum(types.values()) == 2


def test_normalize_adur_output_invalid_spans():
    raw = {"adus": [{"id": "bad", "text": "x", "start": "notint", "end": "no"}]}
    # The adapter currently logs invalid span values and will still return a
    # normalized ADU (with positions possibly empty) rather than raising.
    out = normalize_adur_output(raw, source_text="x")
    assert out["statistics"]["total_adus"] == 1


def test_normalize_are_output_happy_path():
    raw = {
        "adus": [
            {"id": "a1", "text": "A", "label": "claim", "start": 0, "end": 1},
            {"id": "a2", "text": "B", "label": "premise", "start": 2, "end": 3},
        ],
        "relations": [{"id": "r1", "head": "A", "tail": "B", "label": "support"}],
    }
    out = normalize_are_output(raw, source_text="A B")
    assert out["statistics"]["total_adus"] == 2
    assert out["statistics"]["total_relations"] == 1


def test_validate_pipeline_missing_dir(tmp_path: Path):
    # create an empty directory (no model files)
    d = tmp_path / "model_empty"
    d.mkdir()
    ok, details = registry.validate_pipeline({"dir": str(d)})
    assert ok is False
    assert "no expected model config" in details.get("reason", "").lower() or "not found" in details.get("reason", "").lower()


def test_validate_pipeline_with_taskmodule_config(tmp_path: Path):
    d = tmp_path / "model_full"
    d.mkdir()
    # create a fake taskmodule_config.json
    (d / "taskmodule_config.json").write_text(json.dumps({"taskmodule_type": "adu"}))
    ok, details = registry.validate_pipeline({"dir": str(d)})
    assert ok is True
    assert "taskmodule_config.json" in " ".join(details.get("found_files", []))


def test_file_mode_end_to_end_adur_monkeypatched(tmp_path: Path, monkeypatch):
    # Create a small input file
    inp = tmp_path / "doc1.txt"
    inp.write_text("The model performs well. Because it was trained on more data.")

    # Dummy pipeline that mimics the real pipeline.generate behavior
    class DummyPipeline:
        def generate(self, input_path):
            text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
            return {
                "adus": [
                    {"id": "a1", "text": "The model performs well.", "label": "claim", "start": 0, "end": 23, "score": 0.98},
                ]
            }

    # Monkeypatch the registry factory to return our dummy pipeline
    monkeypatch.setattr(registry, "get_adur_pipeline", lambda model_ref=None, **kwargs: DummyPipeline())

    # Run the CLI-level helper
    from moralkg.snowball.phase_1 import cli

    outdir = tmp_path / "out"
    outdir.mkdir()
    res = cli.run_adur_cmd({"dir": "/nonexistent"}, inp, outdir, dry_run=False)

    # run_adur_cmd returns the outdir Path when using file-mode runner
    assert res == outdir

    # Check that the output file exists and has expected schema
    expected_file = outdir / f"adur_{inp.stem}.json"
    assert expected_file.exists()
    data = json.loads(expected_file.read_text(encoding="utf-8"))
    assert "adus" in data and "statistics" in data


def test_file_mode_end_to_end_are_monkeypatched(tmp_path: Path, monkeypatch):
    # Create input file
    inp = tmp_path / "doc2.txt"
    inp.write_text("A B")

    # Dummy ARE pipeline
    class DummyARE:
        def generate(self, input_path):
            return {
                "adus": [
                    {"id": "a1", "text": "A", "label": "claim", "start": 0, "end": 1},
                    {"id": "a2", "text": "B", "label": "premise", "start": 2, "end": 3},
                ],
                "relations": [{"id": "r1", "head": "A", "tail": "B", "label": "support"}],
            }

    monkeypatch.setattr(registry, "get_are_pipeline", lambda model_ref=None, adur_model_ref=None, **kwargs: DummyARE())

    from moralkg.snowball.phase_1 import cli
    outdir = tmp_path / "out_are"
    outdir.mkdir()
    res = cli.run_are_cmd({"dir": "/nonexistent"}, {"dir": "/nonexistent"}, inp, outdir, dry_run=False)
    assert res == outdir
    expected_file = outdir / f"are_{inp.stem}.json"
    assert expected_file.exists()
    data = json.loads(expected_file.read_text(encoding="utf-8"))
    assert data.get("statistics", {}).get("total_relations", 0) == 1 or len(data.get("relations", [])) == 1

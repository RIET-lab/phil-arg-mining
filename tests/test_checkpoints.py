from pathlib import Path
from moralkg.snowball.phase_1.io.checkpoints import save_batch, load_existing, save_checkpoint


def test_save_and_load_batch(tmp_path):
    outdir = tmp_path / "out"
    outputs = [{"id": "p1", "text": "hello"}, {"id": "p2", "text": "world"}]
    p = save_batch(outputs, outdir, "testrun")
    assert p.exists()
    loaded = load_existing(outdir)
    assert any(o.get("id") == "p1" for o in loaded)


def test_save_checkpoint_and_filter(tmp_path):
    outdir = tmp_path / "ck"
    outputs = [{"id": "p3", "text": "x"}]
    ck = save_checkpoint(outputs, outdir, strategy="s1", name="c1")
    assert ck.exists()
    loaded = load_existing(outdir, strategy="s1")
    assert any(o.get("id") == "p3" for o in loaded)

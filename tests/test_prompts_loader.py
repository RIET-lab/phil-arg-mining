import tempfile
from pathlib import Path
from moralkg.snowball.phase_1.prompts.loader import load_prompts


def test_load_prompts_zero_shot(tmp_path):
    d = tmp_path / "zero-shot"
    d.mkdir()
    # create a single user prompt
    user = d / "user_prompt.txt"
    user.write_text("Analyze the text and return JSON.")

    configs = load_prompts(d)
    assert len(configs) == 1
    c = configs[0]
    assert c.shot_type == "zero-shot"
    assert c.user_file.name == "user_prompt.txt"
    assert c.system_file is None
    assert c.user_text.startswith("Analyze")


def test_load_prompts_with_systems(tmp_path):
    d = tmp_path / "few-shot"
    d.mkdir()
    (d / "system_prompt_1.txt").write_text("System A")
    (d / "system_prompt_2.txt").write_text("System B")
    (d / "user_prompt.txt").write_text("User prompt")

    configs = load_prompts(d)
    # two system prompts paired with the single user -> 2 configs
    assert len(configs) == 2
    assert {c.variation for c in configs} == {"zs1", "zs2"}

from moralkg.snowball.phase_1.models.registry import create_end2end


def test_create_mock_end2end():
    e = create_end2end(allow_real=False)
    out = e.generate('sys', 'user')
    assert out.get('trace', {}).get('mock') is True


def test_lazy_attempt_real_falls_back():
    # Allow real, but in our test environment the real End2End will likely
    # raise due to missing HF/model files; the registry should fall back to mock.
    e = create_end2end(allow_real=True)
    out = e.generate('s', 'u')
    # Accept either a mock result or a real model output — require basic shape
    assert isinstance(out, dict)
    assert 'text' in out
    assert 'trace' in out

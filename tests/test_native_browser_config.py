import json
from pathlib import Path


def test_browser_uses_native_window_not_fixed_test_viewport():
    config=json.loads(Path('tools/playwright-native.json').read_text())
    assert config['browser']['contextOptions']['viewport'] is None
    assert '--start-maximized' in config['browser']['launchOptions']['args']

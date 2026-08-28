from pathlib import Path

ROOT = Path(__file__).parents[3] / "workspace/src/aichallenge_tools/teleop_manager/src"


def test_physical_keyboard_bridge_publishes_dedicated_episode_markers():
    source = (ROOT / "keyboard_to_joy_node.cpp").read_text()
    assert '"/racing_maneuver/episode_control"' in source
    assert "KEY_F9" in source and '"START"' in source
    assert "KEY_F10" in source and '"STOP"' in source
    assert "KEY_F11" in source and '"DISCARD"' in source


def test_x11_keyboard_bridge_publishes_dedicated_episode_markers():
    source = (ROOT / "keyboard_x11_to_joy_node.cpp").read_text()
    assert '"/racing_maneuver/episode_control"' in source
    assert '"START"' in source and '"STOP"' in source and '"DISCARD"' in source

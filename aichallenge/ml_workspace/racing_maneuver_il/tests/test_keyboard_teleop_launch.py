from pathlib import Path

import yaml

REPO = Path(__file__).parents[4]


def test_keyboard_teleop_input_source_is_forwarded_from_environment_to_leaf_launch():
    run = (REPO / "aichallenge/run_autoware.bash").read_text()
    system = (
        REPO
        / "aichallenge/workspace/src/aichallenge_system/aichallenge_system_launch/launch/aichallenge_system.launch.xml"
    ).read_text()
    submit = (
        REPO
        / "aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch/launch/aichallenge_submit.launch.xml"
    ).read_text()
    reference = (
        REPO
        / "aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch/launch/reference.launch.xml"
    ).read_text()
    assert "TELEOP_INPUT_SOURCE" in run
    for launch in (system, submit, reference):
        assert "joycon_input_source" in launch
    assert '<arg name="input_source" value="$(var joycon_input_source)"/>' in reference


def test_manual_keyboard_teleop_uses_three_mps2_acceleration_scale():
    config = yaml.safe_load(
        (
            REPO
            / "aichallenge/workspace/src/aichallenge_tools/teleop_manager/config/teleop.param.yaml"
        ).read_text()
    )
    assert config["/**"]["ros__parameters"]["speed_scale"] == 3.0

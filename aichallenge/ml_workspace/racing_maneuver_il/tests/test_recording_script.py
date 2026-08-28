from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "record_maneuver_data.bash"


def test_rosbag_output_path_is_not_precreated():
    script = SCRIPT.read_text()
    assert 'mkdir "${OUTPUT_DIR}"' not in script
    assert 'LOCK_DIR="${OUTPUT_ROOT}/.${RECORDING_ID}.lock"' in script
    assert 'mkdir "${LOCK_DIR}"' in script


def test_recorder_requires_and_records_episode_markers_only_for_teleop_owner():
    script = SCRIPT.read_text()
    assert "EPISODE_CONTROL_TOPIC=/racing_maneuver/episode_control" in script
    assert 'if [[ "${COMMAND_OWNER}" == "teleop_manager_node" ]]; then' in script
    assert 'RECORD_TOPICS+=("${EPISODE_CONTROL_TOPIC}")' in script
    assert "EPISODE_MARKERS_REQUIRED_ARG=(--episode-markers-required)" in script


def test_recorder_skips_episode_marker_preflight_for_nonteleop_owner():
    script = SCRIPT.read_text()
    assert (
        "Episode lifecycle markers are required only for teleop_manager_node." in script
    )
    assert "EPISODE_MARKERS_REQUIRED_ARG=()" in script


def test_recorder_derives_truthful_expert_source_from_command_owner():
    script = SCRIPT.read_text()
    assert "teleop_manager_node) EXPERT_SOURCE=manual" in script
    assert "mpc_controller) EXPERT_SOURCE=mpc" in script
    assert '--expert-source "${EXPERT_SOURCE}"' in script

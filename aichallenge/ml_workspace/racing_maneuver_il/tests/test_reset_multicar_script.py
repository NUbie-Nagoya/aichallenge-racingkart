import os
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[4]
SCRIPT = REPO / "aichallenge/utils/reset_multicar_awsim_autoware.bash"


def test_multicar_reset_publishes_awsim_then_calls_each_domain(tmp_path):
    log = tmp_path / "ros2.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ros2 = fake_bin / "ros2"
    fake_ros2.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s|%s\\n\' "$ROS_DOMAIN_ID" "$*" >> "$FAKE_ROS_LOG"\n'
        "if [[ \"$1 $2\" == 'topic info' ]]; then echo 'Type: std_msgs/msg/Empty'; fi\n"
        "if [[ \"$1 $2\" == 'service type' ]]; then echo 'std_srvs/srv/Trigger'; fi\n"
    )
    fake_ros2.chmod(0o755)
    setup = tmp_path / "setup.bash"
    setup.write_text(f'export PATH="{fake_bin}:$PATH"\n')
    setup.chmod(0o755)

    env = os.environ | {
        "ROS_SETUP_FILE": str(setup),
        "VEHICLE_DOMAINS": "1 2 3",
        "FAKE_ROS_LOG": str(log),
    }
    result = subprocess.run(
        [str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == "0|topic info /admin/awsim/reset"
    assert calls[1] == "0|topic pub --once /admin/awsim/reset std_msgs/msg/Empty {}"
    assert [call for call in calls if "service call /set_initial_pose" in call] == [
        "1|service call /set_initial_pose std_srvs/srv/Trigger {}",
        "2|service call /set_initial_pose std_srvs/srv/Trigger {}",
        "3|service call /set_initial_pose std_srvs/srv/Trigger {}",
    ]
    script = SCRIPT.read_text()
    assert "ros2 topic info /admin/awsim/reset | grep" not in script
    assert "ros2 service type /set_initial_pose | grep" not in script

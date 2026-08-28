#!/usr/bin/env bash
# Reset AWSIM and reinitialize every per-vehicle Autoware localization stack.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: reset_multicar_awsim_autoware.bash [--domains "1 2 3 4"]

Environment:
  ROS_SETUP_FILE         ROS workspace setup path
                         (default: /aichallenge/workspace/install/setup.bash)
  VEHICLE_DOMAINS        Space-separated Autoware ROS domains (default: "1 2 3 4")
  RESET_SETTLE_SECONDS   Optional delay after the AWSIM reset (default: 2)

The AWSIM admin reset is published on ROS domain 0. Each vehicle domain then
receives a /set_initial_pose std_srvs/srv/Trigger request.
EOF
}

VEHICLE_DOMAINS=${VEHICLE_DOMAINS:-"1 2 3 4"}
RESET_SETTLE_SECONDS=${RESET_SETTLE_SECONDS:-2}

while (($#)); do
    case "$1" in
        --domains) VEHICLE_DOMAINS=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "${RESET_SETTLE_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "RESET_SETTLE_SECONDS must be a non-negative number." >&2
    exit 2
fi

ROS_SETUP_FILE=${ROS_SETUP_FILE:-/aichallenge/workspace/install/setup.bash}
if [[ ! -f "${ROS_SETUP_FILE}" ]]; then
    echo "ROS setup file does not exist: ${ROS_SETUP_FILE}" >&2
    exit 1
fi
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP_FILE}"
set -u

export ROS_DOMAIN_ID=0
echo "Resetting AWSIM on ROS domain 0..."
AWSIM_RESET_TOPIC_INFO="$(ros2 topic info /admin/awsim/reset)"
if [[ "${AWSIM_RESET_TOPIC_INFO}" != *"Type: std_msgs/msg/Empty"* ]]; then
    echo "AWSIM reset topic is unavailable or has the wrong type." >&2
    exit 1
fi
ros2 topic pub --once /admin/awsim/reset std_msgs/msg/Empty '{}'

if [[ "${RESET_SETTLE_SECONDS}" != "0" && "${RESET_SETTLE_SECONDS}" != "0.0" ]]; then
    echo "Waiting ${RESET_SETTLE_SECONDS}s for AWSIM reset propagation..."
    sleep "${RESET_SETTLE_SECONDS}"
fi

for domain in ${VEHICLE_DOMAINS}; do
    if ! [[ "${domain}" =~ ^[0-9]+$ ]]; then
        echo "Invalid vehicle ROS domain: ${domain}" >&2
        exit 2
    fi
    export ROS_DOMAIN_ID=${domain}
    echo "Resetting Autoware localization on ROS domain ${domain}..."
    INITIAL_POSE_SERVICE_TYPE="$(ros2 service type /set_initial_pose)"
    if [[ "${INITIAL_POSE_SERVICE_TYPE}" != "std_srvs/srv/Trigger" ]]; then
        echo "/set_initial_pose is unavailable or has the wrong type on domain ${domain}." >&2
        exit 1
    fi
    ros2 service call /set_initial_pose std_srvs/srv/Trigger '{}'
done

echo "AWSIM and Autoware reset sequence completed for domains: ${VEHICLE_DOMAINS}"

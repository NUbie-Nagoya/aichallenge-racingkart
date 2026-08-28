#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: record_maneuver_data.bash --ego-vehicle-id ID --scenario-id ID \
  --maneuver-class CLASS --opponent-count N --opponent-behavior NAME [options]

Classes: follow, pass_left, pass_right, side_by_side, abort, recover, free_lap
Options:
  --seed N              Scenario seed (default: 0)
  --notes TEXT          Recording notes
  --output-root PATH    Bag root (default: <workspace>/rawdata)
  --command-owner NODE  Sole expected publisher node for final commands
  --allow-no-v2x       Permit a diagnostic recording without V2X
  --yes                 Skip the final interactive confirmation
EOF
}

EGO_VEHICLE_ID=""
SCENARIO_ID=""
MANEUVER_CLASS=""
OPPONENT_COUNT=""
OPPONENT_BEHAVIOR=""
SEED=0
NOTES=""
ALLOW_NO_V2X=false
ASSUME_YES=false
OUTPUT_ROOT=""
COMMAND_OWNER=""
EXPERT_SOURCE=""
BAG_PID=""
OUTPUT_DIR=""
PENDING_METADATA=""
LOCK_DIR=""
REQUESTED_STOP=false
ABORT_REQUESTED=false

while (($#)); do
    case "$1" in
        --ego-vehicle-id) EGO_VEHICLE_ID=${2:?}; shift 2 ;;
        --scenario-id) SCENARIO_ID=${2:?}; shift 2 ;;
        --maneuver-class) MANEUVER_CLASS=${2:?}; shift 2 ;;
        --opponent-count) OPPONENT_COUNT=${2:?}; shift 2 ;;
        --opponent-behavior) OPPONENT_BEHAVIOR=${2:?}; shift 2 ;;
        --seed) SEED=${2:?}; shift 2 ;;
        --notes) NOTES=${2:?}; shift 2 ;;
        --output-root) OUTPUT_ROOT=${2:?}; shift 2 ;;
        --command-owner) COMMAND_OWNER=${2:?}; shift 2 ;;
        --allow-no-v2x) ALLOW_NO_V2X=true; shift ;;
        --yes) ASSUME_YES=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value_name in EGO_VEHICLE_ID SCENARIO_ID MANEUVER_CLASS OPPONENT_COUNT OPPONENT_BEHAVIOR COMMAND_OWNER; do
    if [[ -z "${!value_name}" ]]; then
        echo "Missing required argument: ${value_name}" >&2
        usage >&2
        exit 2
    fi
done

case "${COMMAND_OWNER}" in
    teleop_manager_node) EXPERT_SOURCE=manual ;;
    mpc_controller) EXPERT_SOURCE=mpc ;;
    *)
        echo "Unsupported command owner for expert recording: ${COMMAND_OWNER}" >&2
        echo "Expected teleop_manager_node or mpc_controller." >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/rawdata}"
RECORDING_ID="$(date -u +%Y%m%d-%H%M%S)-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
OUTPUT_DIR="${OUTPUT_ROOT}/${RECORDING_ID}"

request_stop() {
    if [[ -z "${BAG_PID}" ]]; then
        exit 130
    fi
    REQUESTED_STOP=true
    if kill -0 "${BAG_PID}" 2>/dev/null; then
        echo
        echo "Stopping and finalizing rosbag recorder..."
        kill -INT "${BAG_PID}" 2>/dev/null || true
    fi
}

abort_recording() {
    ABORT_REQUESTED=true
    if [[ -z "${BAG_PID}" ]]; then
        exit 143
    fi
    if kill -0 "${BAG_PID}" 2>/dev/null; then
        echo "Aborting rosbag recorder after TERM..." >&2
        kill -TERM "${BAG_PID}" 2>/dev/null || true
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${BAG_PID}" ]] && kill -0 "${BAG_PID}" 2>/dev/null; then
        if [[ "${ABORT_REQUESTED}" == true ]]; then
            kill -TERM "${BAG_PID}" 2>/dev/null || true
        else
            kill -INT "${BAG_PID}" 2>/dev/null || true
        fi
        wait "${BAG_PID}" 2>/dev/null || true
    fi
    if [[ -n "${PENDING_METADATA}" && -f "${PENDING_METADATA}" ]]; then
        rm -f -- "${PENDING_METADATA}"
    fi
    if [[ -n "${LOCK_DIR}" && -d "${LOCK_DIR}" ]]; then
        rmdir -- "${LOCK_DIR}" 2>/dev/null || true
    fi
    echo "Recording directory: ${OUTPUT_DIR}"
    exit "${status}"
}
trap cleanup EXIT
trap request_stop INT
trap abort_recording TERM

ROS_SETUP_FILE=${ROS_SETUP_FILE:-/aichallenge/workspace/install/setup.bash}
if [[ ! -f "${ROS_SETUP_FILE}" ]]; then
    echo "ROS setup file does not exist: ${ROS_SETUP_FILE}" >&2
    exit 1
fi
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP_FILE}"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

REQUIRED_TOPICS=(
    /clock
    /sensing/lidar/scan
    /localization/kinematic_state
    /localization/acceleration
    /sensing/imu/imu_raw
    /vehicle/status/velocity_status
    /vehicle/status/steering_status
    /vehicle/status/control_mode
    /control/command/control_cmd
    /awsim/status
)
EPISODE_CONTROL_TOPIC=/racing_maneuver/episode_control
OPTIONAL_TOPICS=(
    /localization/pose
    /localization/twist
    /control/command/control_cmd_raw
    /control/command/actuation_cmd
    /joy
    /control/control_mode_request_topic
    /awsim/control_mode_request_topic
    /awsim/state
    /initialpose
)
V2X_TOPIC=/v2x/vehicle_positions
AVAILABLE_TOPICS="$(ros2 topic list)"

topic_available() {
    local wanted=$1 topic
    while IFS= read -r topic; do
        [[ "${topic}" == "${wanted}" ]] && return 0
    done <<<"${AVAILABLE_TOPICS}"
    return 1
}

missing=0
RECORD_TOPICS=()
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Checking essential manual-expert IL topics..."
for topic in "${REQUIRED_TOPICS[@]}"; do
    if topic_available "${topic}"; then
        printf '  [OK]      %s\n' "${topic}"
        RECORD_TOPICS+=("${topic}")
    else
        printf '  [MISSING] %s\n' "${topic}" >&2
        missing=1
    fi
done

if topic_available "${V2X_TOPIC}"; then
    printf '  [OK]      %s\n' "${V2X_TOPIC}"
    RECORD_TOPICS+=("${V2X_TOPIC}")
elif [[ "${ALLOW_NO_V2X}" == true ]]; then
    printf '  [SKIP]    %s (diagnostic --allow-no-v2x)\n' "${V2X_TOPIC}" >&2
else
    printf '  [MISSING] %s\n' "${V2X_TOPIC}" >&2
    missing=1
fi

EPISODE_MARKERS_REQUIRED_ARG=()
if [[ "${COMMAND_OWNER}" == "teleop_manager_node" ]]; then
    if topic_available "${EPISODE_CONTROL_TOPIC}"; then
        printf '  [OK]      %s (START/STOP/DISCARD lifecycle markers)\n' "${EPISODE_CONTROL_TOPIC}"
        RECORD_TOPICS+=("${EPISODE_CONTROL_TOPIC}")
        EPISODE_MARKERS_REQUIRED_ARG=(--episode-markers-required)
    else
        printf '  [MISSING] %s (build/relaunch keyboard teleop before recording)\n' "${EPISODE_CONTROL_TOPIC}" >&2
        missing=1
    fi
else
    echo "  [SKIP]    Episode lifecycle markers are required only for teleop_manager_node."
fi

for topic in "${OPTIONAL_TOPICS[@]}"; do
    if topic_available "${topic}"; then
        RECORD_TOPICS+=("${topic}")
    else
        printf '  [SKIP]    optional topic unavailable: %s\n' "${topic}" >&2
    fi
done

if ((missing)); then
    echo "Refusing to record: required topics are unavailable." >&2
    exit 1
fi

command_snapshot() {
    ros2 topic info /control/command/control_cmd -v
}

assert_expected_command_owner() {
    local snapshot=$1
    if [[ "${snapshot}" != *"Publisher count: 1"* || "${snapshot}" != *"Node name: ${COMMAND_OWNER}"* ]]; then
        echo "Refusing to record: final command topic must have exactly one publisher, ${COMMAND_OWNER}." >&2
        return 1
    fi
}

COMMAND_SNAPSHOT_BEFORE="$(command_snapshot)"
printf '\nFinal command publisher provenance (before):\n%s\n' "${COMMAND_SNAPSHOT_BEFORE}"
assert_expected_command_owner "${COMMAND_SNAPSHOT_BEFORE}"
printf '\nManual expert metadata:\n'
printf '  recording_id:      %s\n' "${RECORDING_ID}"
printf '  ego_vehicle_id:    %s\n' "${EGO_VEHICLE_ID}"
printf '  scenario_id:       %s\n' "${SCENARIO_ID}"
printf '  maneuver_class:    %s\n' "${MANEUVER_CLASS}"
printf '  opponent_count:    %s\n' "${OPPONENT_COUNT}"
printf '  opponent_behavior: %s\n' "${OPPONENT_BEHAVIOR}"
printf '  seed:               %s\n' "${SEED}"

if [[ "${ASSUME_YES}" != true ]]; then
    read -r -p "Confirm manual mode, command owner, and scenario metadata [y/N]: " answer
    [[ "${answer}" == y || "${answer}" == Y ]] || { echo "Cancelled."; exit 1; }
fi

mkdir -p "${OUTPUT_ROOT}"
LOCK_DIR="${OUTPUT_ROOT}/.${RECORDING_ID}.lock"
if [[ -e "${OUTPUT_DIR}" ]] || ! mkdir "${LOCK_DIR}"; then
    echo "Refusing to record: output path or lock is already reserved: ${OUTPUT_DIR}" >&2
    exit 1
fi
PENDING_METADATA="${OUTPUT_DIR}/.recording.json.pending"
cd "${SCRIPT_DIR}"

printf '\nRecording to %s\n' "${OUTPUT_DIR}"
printf '  %s\n' "${RECORD_TOPICS[@]}"
echo "Press Ctrl+C to stop and finalize."
ros2 bag record "${RECORD_TOPICS[@]}" -o "${OUTPUT_DIR}" -s mcap \
    --compression-format zstd --compression-mode file &
BAG_PID=$!

record_status=0
while true; do
    if wait "${BAG_PID}"; then
        break
    else
        wait_status=$?
    fi
    if [[ "${REQUESTED_STOP}" == true ]] && kill -0 "${BAG_PID}" 2>/dev/null; then
        # The wait was interrupted by the shell trap while rosbag is still
        # finalizing. Re-enter wait and collect the recorder's actual status.
        continue
    fi
    if [[ "${ABORT_REQUESTED}" == true ]]; then
        record_status=143
    elif [[ "${REQUESTED_STOP}" == true && "${wait_status}" == 130 ]]; then
        # rosbag commonly exits 130 after the requested SIGINT. Finalization is
        # proven below by metadata.yaml and ros2 bag info, not by that code.
        record_status=0
    else
        record_status=${wait_status}
    fi
    break
done
BAG_PID=""
if ((record_status)); then
    echo "rosbag recorder exited with status ${record_status}; bag is invalid." >&2
    exit "${record_status}"
fi
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Recording output directory was not created." >&2
    exit 1
fi
if [[ ! -f "${OUTPUT_DIR}/metadata.yaml" ]]; then
    echo "Recording did not finalize: metadata.yaml is absent." >&2
    exit 1
fi
COMMAND_SNAPSHOT_AFTER="$(command_snapshot)"
printf '\nFinal command publisher provenance (after):\n%s\n' "${COMMAND_SNAPSHOT_AFTER}"
assert_expected_command_owner "${COMMAND_SNAPSHOT_AFTER}"
V2X_REQUIRED_ARG=()
if [[ "${ALLOW_NO_V2X}" == false ]]; then
    V2X_REQUIRED_ARG=(--v2x-required)
fi
PYTHONPATH="${SCRIPT_DIR}" python3 scripts/write_recording_metadata.py \
    --output "${PENDING_METADATA}" \
    --recording-id "${RECORDING_ID}" \
    --ego-vehicle-id "${EGO_VEHICLE_ID}" \
    --scenario-id "${SCENARIO_ID}" \
    --maneuver-class "${MANEUVER_CLASS}" \
    --opponent-count "${OPPONENT_COUNT}" \
    --opponent-behavior "${OPPONENT_BEHAVIOR}" \
    --seed "${SEED}" \
    --notes "${NOTES}" \
    --ros-domain-id "${ROS_DOMAIN_ID}" \
    --command-owner "${COMMAND_OWNER}" \
    --expert-source "${EXPERT_SOURCE}" \
    --command-snapshot-before "${COMMAND_SNAPSHOT_BEFORE}" \
    --command-snapshot-after "${COMMAND_SNAPSHOT_AFTER}" \
    "${EPISODE_MARKERS_REQUIRED_ARG[@]}" \
    "${V2X_REQUIRED_ARG[@]}"
mv "${PENDING_METADATA}" "${OUTPUT_DIR}/recording.json"
PENDING_METADATA=""
if [[ ! -f "${OUTPUT_DIR}/recording.json" ]]; then
    echo "Recording metadata could not be finalized." >&2
    exit 1
fi
ros2 bag info "${OUTPUT_DIR}"
echo "Recording finalized. Run inventory_bags.py before extraction."

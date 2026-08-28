#!/usr/bin/env bash
set -euo pipefail

NODE_NAME=${RACING_MANEUVER_NODE_NAME:-/racing_maneuver_controller}
DEBUG_COMMAND=${RACING_MANEUVER_DEBUG_COMMAND_TOPIC:-/racing_maneuver/debug/control_cmd}
DEBUG_FEATURES=${RACING_MANEUVER_DEBUG_FEATURES_TOPIC:-/racing_maneuver/debug/features}
DEBUG_SAFETY=${RACING_MANEUVER_DEBUG_SAFETY_TOPIC:-/racing_maneuver/debug/safety_status}
REAL_COMMAND=${RACING_MANEUVER_REAL_COMMAND_TOPIC:-/control/command/control_cmd}
SAMPLE_TIMEOUT=${SAMPLE_TIMEOUT:-8}

printf '== Effective safety parameters ==\n'
ros2 param get "${NODE_NAME}" publish_commands
ros2 param get "${NODE_NAME}" model_path

publish_value="$(ros2 param get "${NODE_NAME}" publish_commands)"
if [[ "${publish_value}" != *"False"* && "${publish_value}" != *"false"* ]]; then
    echo "Refusing shadow validation: publish_commands is not false." >&2
    exit 1
fi

printf '\n== Endpoint ownership (not delivery proof) ==\n'
ros2 topic info "${REAL_COMMAND}" -v

printf '\n== One debug command ==\n'
timeout "${SAMPLE_TIMEOUT}" ros2 topic echo --once "${DEBUG_COMMAND}"

printf '\n== One feature vector ==\n'
timeout "${SAMPLE_TIMEOUT}" ros2 topic echo --once "${DEBUG_FEATURES}"

printf '\n== One safety status ==\n'
timeout "${SAMPLE_TIMEOUT}" ros2 topic echo --once "${DEBUG_SAFETY}"

cat <<EOF

Shadow samples received. Measure command rate separately for at least 10 seconds:
  timeout 12 ros2 topic hz ${DEBUG_COMMAND}

This script proves debug visibility and a non-actuating effective parameter only.
It does not prove maneuver safety or that the real command topic is silent; record
and inspect the real command stream with publisher GIDs during the scenario.
EOF

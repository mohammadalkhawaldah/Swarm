#!/usr/bin/env bash
# Launch all SITL + MAVSDK servers in a dedicated tmux session.
# Each SITL command runs in its own pane, followed by its MAVSDK server.
# Close the tmux session (prefix + :) or kill individual panes to stop components.

set -euo pipefail

SESSION_NAME="swarm"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n "uav1" \
    "sim_vehicle.py -v ArduCopter --console --map --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14540 -L CMAC --add-param-file=battery_config.parm"
tmux split-window -v -t "$SESSION_NAME:0" \
    "mavsdk_server -p 50040 udp://:14540"

tmux new-window -t "$SESSION_NAME" -n "uav2" \
    "sim_vehicle.py -v ArduCopter -I 1 --sysid 2 --console --map --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14541 -L CMAC_100M_EAST --add-param-file=battery_config.parm"
tmux split-window -v -t "$SESSION_NAME:1" \
    "mavsdk_server -p 50041 udp://:14541"

tmux new-window -t "$SESSION_NAME" -n "uav3" \
    "sim_vehicle.py -v ArduCopter -I 2 --sysid 3 --console --map --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14542 -L SITL_3_LOCATION --add-param-file=battery_config.parm"
tmux split-window -v -t "$SESSION_NAME:2" \
    "mavsdk_server -p 50042 udp://:14542"

tmux new-window -t "$SESSION_NAME" -n "uav4" \
    "sim_vehicle.py -v ArduCopter -I 3 --sysid 4 --console --map --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14543 -L SITL_4_LOCATION --add-param-file=battery_config.parm"
tmux split-window -v -t "$SESSION_NAME:3" \
    "mavsdk_server -p 50043 udp://:14543"

tmux new-window -t "$SESSION_NAME" -n "uav5" \
    "sim_vehicle.py -v ArduCopter -I 4 --sysid 5 --console --map --out=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14544 -L SITL_5_LOCATION --add-param-file=battery_config.parm"
tmux split-window -v -t "$SESSION_NAME:4" \
    "mavsdk_server -p 50044 udp://:14544"

echo "tmux session '$SESSION_NAME' is ready. Attach with: tmux attach -t $SESSION_NAME"

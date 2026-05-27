#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script expects root privileges on the AutoDL Ubuntu container." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y software-properties-common curl gnupg lsb-release locales
locale-gen en_US en_US.UTF-8 || true
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 || true

add-apt-repository -y universe
apt-get update
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
  > /etc/apt/sources.list.d/ros2.list
apt-get update
apt-get install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep

source /opt/ros/humble/setup.bash
ros2 --help >/dev/null
python - <<'PY'
import rclpy
import std_msgs
print("rclpy and std_msgs are importable")
PY

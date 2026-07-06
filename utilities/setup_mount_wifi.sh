#!/usr/bin/env bash
# setup_mount_wifi.sh — Configure wlan-mount to auto-connect to the telescope mount's WiFi.
#
# The ZWO AM5 (and similar mounts) act as their own WiFi access point.
# This script tells NetworkManager to connect wlan-mount to that network
# automatically whenever the mount is powered on.
#
# Prerequisites:
#   - Run install.py first with both Edimax adapters plugged in
#   - Reboot the Pi so the wlan-mount interface name is active
#
# Run once as root:
#   sudo bash utilities/setup_mount_wifi.sh
#
# Edit SSID and PASSWORD below to match your mount's WiFi settings.
# For the ZWO AM5, find these in the ZWO app under Network > Hotspot settings.

set -euo pipefail

SSID="ZWO_AM5_XXXXXX"    # Replace with your mount's WiFi network name
PASSWORD="12345678"        # Replace with your mount's WiFi password
BSSID=""                   # Optional: set to mount's MAC (e.g. "7C:87:CE:5C:81:B1") for faster/more reliable connection
INTERFACE="wlan-mount"
CONNECTION_NAME="mount-wifi"

# Static IP for the Pi on the mount's subnet.
# The ZWO AM5 is always the gateway at 192.168.4.1 in hotspot mode.
# A static IP avoids relying on the mount's DHCP server, which can be slow.
PI_IP="192.168.4.100/24"
GATEWAY="192.168.4.1"

if [[ $EUID -ne 0 ]]; then
    echo "Run this script as root:  sudo bash $0"
    exit 1
fi

# Verify wlan-mount exists — requires install.py to have run and Pi rebooted
if ! ip link show "$INTERFACE" &>/dev/null; then
    echo "Error: interface '$INTERFACE' not found."
    echo "Make sure you ran install.py with both Edimax adapters plugged in and rebooted."
    exit 1
fi

# Remove existing profile if re-running
if nmcli connection show "$CONNECTION_NAME" &>/dev/null; then
    echo "Removing existing '$CONNECTION_NAME' connection profile..."
    nmcli connection delete "$CONNECTION_NAME"
fi

echo "Configuring $INTERFACE to auto-connect to '$SSID'..."
NM_ARGS=(
    type wifi
    ifname "$INTERFACE"
    con-name "$CONNECTION_NAME"
    ssid "$SSID"
    wifi-sec.key-mgmt wpa-psk
    wifi-sec.psk "$PASSWORD"
    ipv4.method manual
    ipv4.addresses "$PI_IP"
    ipv4.gateway "$GATEWAY"
    ipv6.method disabled
    connection.autoconnect yes
)
if [[ -n "$BSSID" ]]; then
    NM_ARGS+=(wifi.bssid "$BSSID")
fi
nmcli connection add "${NM_ARGS[@]}"

echo ""
echo "Done."
echo "  Interface : $INTERFACE"
echo "  Network   : $SSID"
[[ -n "$BSSID" ]] && echo "  BSSID     : $BSSID"
echo ""
echo "wlan-mount will connect to the mount automatically whenever it is powered on."
echo ""
echo "To connect now (if the mount is already powered on):"
echo "  sudo nmcli connection up mount-wifi"

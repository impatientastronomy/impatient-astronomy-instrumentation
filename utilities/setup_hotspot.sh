#!/usr/bin/env bash
# setup_hotspot.sh — One-time Raspberry Pi guest hotspot configuration.
#
# Creates a concurrent WiFi access point (uap0) on the Pi's built-in radio
# alongside the existing home-network connection on wlan0.  wlan0 stays on
# the home network throughout; uap0 broadcasts the AstroEye guest hotspot.
#
# The hotspot does NOT start automatically on boot.  The eyepiece app
# launches it on startup and tears it down on exit.
# SSH over the home network is unaffected.
#
# Prerequisites: Pi OS Bookworm.  Run after completing the rest of install.
#
# Run once as root:
#   sudo bash utilities/setup_hotspot.sh
#
# Edit SSID / PASSWORD / IP / PORT below to match your configuration.yaml
# hotspot section if you changed the defaults there.

set -euo pipefail

SSID="AstroEye"
PASSWORD="stargazer"
AP_IFACE="uap0"
STA_IFACE="wlan0"
IP="192.168.10.1"
CONNECTION_NAME="astro-hotspot"
PORT=8080
APP_USER="${SUDO_USER:-pi}"

if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root:  sudo bash $0"
  exit 1
fi

if ! ip link show "$STA_IFACE" &>/dev/null; then
    echo "Error: $STA_IFACE not found."
    exit 1
fi

# Remove existing NM profile if re-running
if nmcli connection show "$CONNECTION_NAME" &>/dev/null; then
    echo "Removing existing '$CONNECTION_NAME' profile..."
    nmcli connection delete "$CONNECTION_NAME"
fi

echo "Creating hotspot NM profile (interface: $AP_IFACE)..."
nmcli connection add \
    type wifi \
    ifname "$AP_IFACE" \
    con-name "$CONNECTION_NAME" \
    ssid "$SSID" \
    mode ap \
    wifi.band bg \
    ipv4.method shared \
    ipv4.addresses "${IP}/24" \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASSWORD" \
    connection.autoconnect no

# DNS redirect: all hostnames resolve to Pi so captive portal triggers on
# iOS/Android automatically, popping up the gallery page.
mkdir -p /etc/NetworkManager/dnsmasq-shared.d
printf 'address=/#/%s\n' "$IP" > /etc/NetworkManager/dnsmasq-shared.d/captive-portal.conf

# Redirect HTTP (port 80) to gallery port so the gallery serves as the
# captive portal page phones display after joining.
echo "Setting up port 80 → ${PORT} redirect..."
apt-get install -y iptables-persistent
iptables -t nat -D PREROUTING -i "$AP_IFACE" -p tcp --dport 80 \
    -j DNAT --to-destination "${IP}:${PORT}" 2>/dev/null || true
iptables -t nat -A PREROUTING -i "$AP_IFACE" -p tcp --dport 80 \
    -j DNAT --to-destination "${IP}:${PORT}"
netfilter-persistent save

# Helper scripts invoked by the app — installed to /usr/local/bin so the
# sudoers entry can reference them by exact path.
printf '#!/bin/bash\nif ! ip link show %s &>/dev/null; then\n    iw dev %s interface add %s type __ap\nfi\nif nmcli connection show --active %s &>/dev/null; then\n    exit 0\nfi\nfor i in $(seq 10); do\n    nmcli device status 2>/dev/null | grep -q "^%s.*disconnected" && break\n    sleep 0.5\ndone\nnmcli connection up %s\n' \
    "$AP_IFACE" "$STA_IFACE" "$AP_IFACE" "$CONNECTION_NAME" "$AP_IFACE" "$CONNECTION_NAME" \
    > /usr/local/bin/astro-hotspot-start
chmod +x /usr/local/bin/astro-hotspot-start

printf '#!/bin/bash\nnmcli connection down %s 2>/dev/null || true\niw dev %s del 2>/dev/null || true\n' \
    "$CONNECTION_NAME" "$AP_IFACE" \
    > /usr/local/bin/astro-hotspot-stop
chmod +x /usr/local/bin/astro-hotspot-stop

# Passwordless sudo for the app user — scoped to only these two scripts.
printf '%s ALL=(ALL) NOPASSWD: /usr/local/bin/astro-hotspot-start\n' "$APP_USER" \
    > /etc/sudoers.d/astro-hotspot
printf '%s ALL=(ALL) NOPASSWD: /usr/local/bin/astro-hotspot-stop\n'  "$APP_USER" \
    >> /etc/sudoers.d/astro-hotspot
chmod 440 /etc/sudoers.d/astro-hotspot

echo ""
echo "=== Done ==="
echo "  Hotspot SSID : $SSID"
echo "  Password     : $PASSWORD"
echo "  Gallery URL  : http://${IP}:${PORT}"
echo ""
echo "The hotspot launches automatically when the app starts."
echo "It shuts down when the app exits.  wlan0 and SSH are unaffected."

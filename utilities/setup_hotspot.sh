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
LOG_FILE=/var/log/astro-hotspot.log
touch "$LOG_FILE"
chmod 644 "$LOG_FILE"   # root writes it; user can read it

cat > /usr/local/bin/astro-hotspot-start << SCRIPT
#!/bin/bash
# All output appended to a persistent trace log for debugging.
LOG=/var/log/astro-hotspot.log
exec >> "\$LOG" 2>&1
echo "[\$(date)] --- astro-hotspot-start invoked (pid=\$\$) ---"

# Wait for the STA interface to finish connecting — the Broadcom radio
# can't bring up the AP while it is mid-authentication on wlan0.
echo "[\$(date)] waiting for ${STA_IFACE} connected..."
for i in \$(seq 30); do
    nmcli device status 2>/dev/null | grep -q "^${STA_IFACE}.*connected" && break
    sleep 1
done
echo "[\$(date)] ${STA_IFACE} state: \$(nmcli -t -f STATE device show ${STA_IFACE} 2>/dev/null)"

# Create uap0 virtual AP interface if it doesn't exist.
if ! ip link show ${AP_IFACE} &>/dev/null; then
    echo "[\$(date)] creating ${AP_IFACE}..."
    iw dev ${STA_IFACE} interface add ${AP_IFACE} type __ap
fi

# If NM reports the connection as active but the interface isn't actually
# broadcasting (NO-CARRIER or link not UP), tear it down and restart.
if nmcli connection show --active ${CONNECTION_NAME} &>/dev/null; then
    if ip link show ${AP_IFACE} 2>/dev/null | grep -q "LOWER_UP"; then
        echo "[\$(date)] already active and link is UP, exiting."
        exit 0
    fi
    echo "[\$(date)] NM reports active but link is down — restarting."
    nmcli connection down ${CONNECTION_NAME} 2>/dev/null || true
    sleep 1
fi

# Wait for NM to register and initialize uap0 (up to 10 s).
echo "[\$(date)] waiting for ${AP_IFACE} disconnected..."
for i in \$(seq 20); do
    nmcli device status 2>/dev/null | grep -q "^${AP_IFACE}.*disconnected" && break
    sleep 0.5
done
echo "[\$(date)] ${AP_IFACE} device status: \$(nmcli -t -f STATE device show ${AP_IFACE} 2>/dev/null)"

# Give NM's internal supplicant state a moment to settle before activating.
sleep 1

# Activate with retries — NM occasionally needs a second attempt.
for attempt in 1 2 3; do
    echo "[\$(date)] attempt \$attempt: nmcli connection up ${CONNECTION_NAME}"
    nmcli connection up ${CONNECTION_NAME} || true
    sleep 1
    if ip link show ${AP_IFACE} 2>/dev/null | grep -q "LOWER_UP"; then
        echo "[\$(date)] success — ${AP_IFACE} is UP."
        exit 0
    fi
    sleep 2
done

echo "[\$(date)] all attempts failed."
exit 1
SCRIPT
chmod +x /usr/local/bin/astro-hotspot-start

cat > /usr/local/bin/astro-hotspot-stop << SCRIPT
#!/bin/bash
nmcli connection down ${CONNECTION_NAME} 2>/dev/null || true
iw dev ${AP_IFACE} del 2>/dev/null || true
SCRIPT
chmod +x /usr/local/bin/astro-hotspot-stop

# Passwordless sudo for the app user — scoped to only these two scripts.
# !requiretty lets sudo run these scripts from a non-TTY context (e.g. Popen).
{
    printf '# astro-hotspot: allow %s to start/stop the guest WiFi hotspot\n' "$APP_USER"
    printf 'Defaults!/usr/local/bin/astro-hotspot-start !requiretty\n'
    printf 'Defaults!/usr/local/bin/astro-hotspot-stop  !requiretty\n'
    printf '%s ALL=(ALL) NOPASSWD: /usr/local/bin/astro-hotspot-start\n' "$APP_USER"
    printf '%s ALL=(ALL) NOPASSWD: /usr/local/bin/astro-hotspot-stop\n'  "$APP_USER"
} > /etc/sudoers.d/astro-hotspot
chmod 440 /etc/sudoers.d/astro-hotspot

echo ""
echo "=== Done ==="
echo "  Hotspot SSID : $SSID"
echo "  Password     : $PASSWORD"
echo "  Gallery URL  : http://${IP}:${PORT}"
echo ""
echo "The hotspot launches automatically when the app starts."
echo "It shuts down when the app exits.  wlan0 and SSH are unaffected."

#!/usr/bin/env bash
# setup_hotspot.sh — One-time Raspberry Pi hotspot configuration.
#
# Creates a WiFi access point on wlan-guest (USB dongle, pinned by udev rule)
# so guests can join and access the AstroEye image gallery without needing
# the internet.  wlan0 remains free for SSH/setup; wlan-mount connects to
# the mount's WiFi network.
# Run install.py first to set up the wlan-guest udev name rule.
#
# Run once as root:
#   sudo bash utilities/setup_hotspot.sh
#
# Edit SSID / PASSWORD / IP below before running if you changed them
# in configuration.yaml.
#
# Tested on Raspberry Pi OS Bookworm (64-bit).

set -euo pipefail

SSID="AstroEye"
PASSWORD="stargazer"
INTERFACE="wlan-guest"
IP="192.168.10.1"
DHCP_RANGE="192.168.10.10,192.168.10.50,12h"
PORT=8080

if [[ $EUID -ne 0 ]]; then
  echo "Run this script as root:  sudo bash $0"
  exit 1
fi

echo "=== Installing hostapd and dnsmasq ==="
apt-get install -y hostapd dnsmasq

echo "=== Setting static IP on $INTERFACE ==="
cat >> /etc/dhcpcd.conf << EOF

# AstroEye hotspot (added by setup_hotspot.sh)
interface $INTERFACE
    static ip_address=${IP}/24
    nohook wpa_supplicant
EOF

echo "=== Configuring dnsmasq ==="
cat > /etc/dnsmasq.d/astro-hotspot.conf << EOF
interface=$INTERFACE
dhcp-range=$DHCP_RANGE
# Redirect all DNS to Pi so captive portal triggers on any domain
address=/#/$IP
EOF

echo "=== Configuring hostapd ==="
cat > /etc/hostapd/astro-hotspot.conf << EOF
interface=$INTERFACE
driver=nl80211
ssid=$SSID
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=$PASSWORD
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

# Point hostapd at our config
sed -i 's|#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/astro-hotspot.conf"|' \
    /etc/default/hostapd

echo "=== Setting up iptables captive portal redirect ==="
# Redirect HTTP (port 80) from hotspot clients to our gallery server
iptables -t nat -A PREROUTING -i "$INTERFACE" -p tcp --dport 80 \
    -j DNAT --to-destination "${IP}:${PORT}"
# Persist rules across reboots
apt-get install -y iptables-persistent
netfilter-persistent save

echo "=== Enabling services ==="
systemctl unmask hostapd
systemctl enable hostapd
systemctl enable dnsmasq
systemctl restart dhcpcd
systemctl start hostapd
systemctl start dnsmasq

echo ""
echo "=== Done ==="
echo "Hotspot SSID : $SSID"
echo "Password     : $PASSWORD"
echo "Gallery URL  : http://${IP}:${PORT}"
echo ""
echo "Reboot recommended:  sudo reboot"
echo ""
echo "To generate the printable QR code:"
echo "  uv run python utilities/print_qr.py"

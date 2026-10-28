# Legacy Firecracker Setup Notes

Warning: This runbook is not verified or part of the supported `khalamari` service workflow. The commands change host
networking and delete files. Review each command before you use it.

The repository keeps these notes as a historical reference. The code blocks remain unchanged.

## 1. Install Lima and the Ubuntu host

```
brew install lima
limactl start --set '.nestedVirtualization=true' --name=mvm template://default
```

## 2. Open the Linux shell

```
limactl shell mvm
```

## 3. Install Firecracker

Refer to the [Firecracker releases](https://github.com/firecracker-microvm/firecracker/releases).

```
cd /tmp
FC_VERSION="v1.14.1"
ARCH="aarch64"  # Because we're on Apple Silicon

wget https://github.com/firecracker-microvm/firecracker/releases/download/${FC_VERSION}/firecracker-${FC_VERSION}-${ARCH}.tgz
tar -xzf firecracker-${FC_VERSION}-${ARCH}.tgz
sudo mv release-${FC_VERSION}-${ARCH}/firecracker-${FC_VERSION}-${ARCH} /usr/local/bin/firecracker
sudo chmod +x /usr/local/bin/firecracker
```

## 4. Download the Ubuntu guest

```
FC_VERSION="v1.14" # not v1.14.1
ARCH="aarch64"

# Grab the latest kernel
latest_kernel_key=$(wget "http://spec.ccfc.min.s3.amazonaws.com/?prefix=firecracker-ci/${FC_VERSION}/${ARCH}/vmlinux-5.10&list-type=2" -O - 2>/dev/null | grep "(?<=<Key>)(firecracker-ci/${FC_VERSION}/${ARCH}/vmlinux-5\.10\.[0-9]{3})(?=</Key>)" -o -P)
wget "https://s3.amazonaws.com/spec.ccfc.min/${latest_kernel_key}"

# Get Ubuntu rootfs
latest_ubuntu_key=$(curl "http://spec.ccfc.min.s3.amazonaws.com/?prefix=firecracker-ci/${FC_VERSION}/${ARCH}/ubuntu-&list-type=2" | grep -oP "(?<=<Key>)(firecracker-ci/${FC_VERSION}/${ARCH}/ubuntu-[0-9]+\.[0-9]+\.squashfs)(?=</Key>)" | sort -V | tail -1)
ubuntu_version=$(basename $latest_ubuntu_key .squashfs | grep -oE '[0-9]+\.[0-9]+')
wget -O ubuntu-$ubuntu_version.squashfs.upstream "https://s3.amazonaws.com/spec.ccfc.min/$latest_ubuntu_key"
```

## 5. Convert SquashFS to ext4

```convert-squashfs-to-ext4.sh
#!/usr/bin/env bash
# convert-squashfs-to-ext4.sh
set -euo pipefail

SQUASHFS="${1:-ubuntu-24.04.squashfs.upstream}"
EXT4="${2:-ubuntu-24.04.ext4}"
SIZE_MB="${3:-2048}"
KEY_BASENAME="${4:-ubuntu-24.04.id_rsa}"   # will create KEY_BASENAME + .pub if missing

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }

# deps (Ubuntu/Debian)
sudo apt-get update -y
sudo apt-get install -y squashfs-tools e2fsprogs openssh-client coreutils util-linux

need_cmd unsquashfs
need_cmd mkfs.ext4
need_cmd ssh-keygen

# sanity
[[ -f "$SQUASHFS" ]] || { echo "no squashfs: $SQUASHFS"; exit 1; }

# 1) make ext4
rm -f "$EXT4"
dd if=/dev/zero of="$EXT4" bs=1M count="$SIZE_MB" status=progress
mkfs.ext4 -F "$EXT4" >/dev/null

# 2) unpack squashfs into ext4
mnt="$(mktemp -d)"
cleanup() { sudo umount "$mnt" 2>/dev/null || true; rmdir "$mnt" 2>/dev/null || true; }
trap cleanup EXIT

sudo mount -o loop "$EXT4" "$mnt"
sudo unsquashfs -f -d "$mnt" "$SQUASHFS"

# 3) ensure root ssh key (so your start-vm.sh can ssh in)
if [[ ! -f "$KEY_BASENAME" ]]; then
  ssh-keygen -t ed25519 -N '' -f "$KEY_BASENAME"
fi

sudo mkdir -p "$mnt/root/.ssh"
sudo sh -c "cat '${KEY_BASENAME}.pub' >> '$mnt/root/.ssh/authorized_keys'"
sudo chmod 700 "$mnt/root/.ssh"
sudo chmod 600 "$mnt/root/.ssh/authorized_keys"
sudo chown -R root:root "$mnt/root/.ssh"

sync
sudo umount "$mnt"

echo "OK:"
echo "  ext4: $EXT4"
echo "  ssh key: $KEY_BASENAME (and .pub)"
echo "Next: restart firecracker, then run ./start-vm.sh"
```

## 6. Start Firecracker

```
sudo rm -f /tmp/firecracker.socket
sudo firecracker --api-sock /tmp/firecracker.socket --enable-pci
```

## 7. Start the guest

```start-vm.sh
#!/bin/bash

TAP_DEV="tap0"
TAP_IP="172.16.0.1"
MASK_SHORT="/30"

# Setup network interface
sudo ip link del "$TAP_DEV" 2> /dev/null || true
sudo ip tuntap add dev "$TAP_DEV" mode tap
sudo ip addr add "${TAP_IP}${MASK_SHORT}" dev "$TAP_DEV"
sudo ip link set dev "$TAP_DEV" up

# Enable ip forwarding
sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
sudo iptables -P FORWARD ACCEPT

# This tries to determine the name of the host network interface to forward
# VM's outbound network traffic through. If outbound traffic doesn't work,
# double check this returns the correct interface!
HOST_IFACE=$(ip -j route list default |jq -r '.[0].dev')

# Set up microVM internet access
sudo iptables -t nat -D POSTROUTING -o "$HOST_IFACE" -j MASQUERADE 2>/dev/null || true
sudo iptables -t nat -A POSTROUTING -o "$HOST_IFACE" -j MASQUERADE

API_SOCKET="/tmp/firecracker.socket"
LOGFILE="./firecracker.log"

# Create log file
touch $LOGFILE

# Set log file
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"log_path\": \"${LOGFILE}\",
        \"level\": \"Debug\",
        \"show_level\": true,
        \"show_log_origin\": true
    }" \
    "http://localhost/logger"

KERNEL="./$(ls vmlinux* | tail -1)"
KERNEL_BOOT_ARGS="console=ttyS0 reboot=k panic=1"

ARCH=$(uname -m)

if [ ${ARCH} = "aarch64" ]; then
    KERNEL_BOOT_ARGS="keep_bootcon ${KERNEL_BOOT_ARGS}"
fi

# Set boot source
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"kernel_image_path\": \"${KERNEL}\",
        \"boot_args\": \"${KERNEL_BOOT_ARGS}\"
    }" \
    "http://localhost/boot-source"

ROOTFS="./$(ls *.ext4 | tail -1)"

# Set rootfs
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"drive_id\": \"rootfs\",
        \"path_on_host\": \"${ROOTFS}\",
        \"is_root_device\": true,
        \"is_read_only\": false
    }" \
    "http://localhost/drives/rootfs"

# The IP address of a guest is derived from its MAC address with
# `fcnet-setup.sh`, this has been pre-configured in the guest rootfs. It is
# important that `TAP_IP` and `FC_MAC` match this.
FC_MAC="06:00:AC:10:00:02"

# Set network interface
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"iface_id\": \"net1\",
        \"guest_mac\": \"$FC_MAC\",
        \"host_dev_name\": \"$TAP_DEV\"
    }" \
    "http://localhost/network-interfaces/net1"

# API requests are handled asynchronously, it is important the configuration is
# set, before `InstanceStart`.
sleep 0.015s

# Start microVM
sudo curl -X PUT --unix-socket "${API_SOCKET}" \
    --data "{
        \"action_type\": \"InstanceStart\"
    }" \
    "http://localhost/actions"

# API requests are handled asynchronously, it is important the microVM has been
# started before we attempt to SSH into it.
echo "Waiting for microVM to start..."
sleep 5s

KEY_NAME=./$(ls *.id_rsa | tail -1)

# Wait for SSH to be available
echo "Waiting for SSH to be available..."
for i in {1..30}; do
    if ssh -i $KEY_NAME -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@172.16.0.2 "echo 'SSH is ready'" >/dev/null 2>&1; then
        echo "SSH is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Warning: SSH not available after 30 seconds, continuing anyway..."
    fi
    sleep 1
done

# Setup internet access in the guest
echo "Setting up internet access in the guest..."
ssh -i $KEY_NAME -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@172.16.0.2 "ip route add default via 172.16.0.1 dev eth0" || echo "Warning: Failed to set up internet access"

# Setup DNS resolution in the guest
echo "Setting up DNS resolution in the guest..."
ssh -i $KEY_NAME -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@172.16.0.2 "echo 'nameserver 8.8.8.8' > /etc/resolv.conf" || echo "Warning: Failed to set up DNS"

# SSH into the microVM
echo "To connect to the microVM, run:"
echo "ssh -i $KEY_NAME root@172.16.0.2"
echo ""
echo "Use 'root' for both the login and password."
echo "Run 'reboot' to exit."
```

## 8. Remove Firecracker

Warning: This command deletes the Firecracker binary from `/usr/local/bin`.

```
rm /usr/local/bin/firecracker
```

## References

- [Running Firecracker microVMs on a MacBook Pro](https://u3n.medium.com/the-future-of-development-is-here-running-firecracker-microvms-on-your-macbook-pro-m3-ad6fd3e5092c)
- [Firecracker Lima VM](https://github.com/yashdiq/firecracker-lima-vm/tree/main)

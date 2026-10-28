# Legacy Firecracker Commands

Warning: These commands are not verified or part of the supported `khalamari` service workflow. Review the
[setup notes](setup.md) and each command before use.

## macOS layer

```
limactl start mvm

limactl stop mvm

limactl edit mvm

limactl shell mvm
```

## Host operating-system layer

Terminal 0:

```
sudo rm -f /tmp/firecracker.socket
sudo firecracker --api-sock /tmp/firecracker.socket --enable-pci
```

Terminal 1:

```
start-vm.sh
```

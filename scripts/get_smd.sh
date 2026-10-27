#!/bin/sh
# Download the Server Machine Dataset (OmniAnomaly release, MIT license) into data/smd
# at a pinned commit, and write checksums for reproducibility.
set -e
COMMIT=7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4
cd "$(dirname "$0")/../data"
rm -rf _omni smd
git clone -q --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git _omni
git -C _omni checkout -q "$COMMIT"
git -C _omni sparse-checkout set ServerMachineDataset
mv _omni/ServerMachineDataset smd && rm -rf _omni
(cd smd && find train test test_label -name '*.txt' | sort | xargs shasum -a 256) > smd_sha256.txt
echo "SMD ready at data/smd (commit $COMMIT)"

#!/bin/sh
# Download the NASA SMAP and MSL telemetry anomaly datasets (Hundman et al., "Detecting
# Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding", KDD 2018;
# https://github.com/khundman/telemanom) into data/nasa, and write/verify checksums.
#
# Source: the original data release referenced by the paper's repository,
#   https://s3-us-west-2.amazonaws.com/telemanom/data.zip
# That bucket now returns 403, so we fetch the byte-identical Internet Archive snapshot of
# that exact URL (pinned below, sha256 checked). labeled_anomalies.csv comes from the
# telemanom repository at a pinned commit. The repository README now points to a Kaggle
# mirror (patrickfleith/nasa-anomaly-detection-dataset-smap-msl); NASA_SOURCE=kaggle uses it
# instead, and the per-file checksums in data/nasa_sha256.txt must still match.
#
# Layout after download (values exactly as released; no re-scaling by us):
#   data/nasa/train/<chan>.npy   (T, F) float64: column 0 = the telemetry value, already
#   data/nasa/test/<chan>.npy    scaled to [-1, 1] by the authors; columns 1.. = one-hot
#                                encoded spacecraft commands (0/1). F = 25 (SMAP), 55 (MSL).
#   data/nasa/labeled_anomalies.csv  chan_id, spacecraft, anomaly_sequences (test indices)
# The train series contain no labeled anomalies; test labels come from anomaly_sequences.
# See coda.data for how channels are used (all F columns are model inputs).
#
# Caveat (cite in the paper): Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks
# are Flawed and are Creating the Illusion of Progress", IEEE TKDE 2023, criticize SMAP/MSL
# (trivial anomalies, mislabeling, run-to-failure bias). CODA measures false-alarm control
# under quantization, which depends mostly on normal data, so this matters less for FAR than
# for recall; still report it.
set -e
ZIP_URL="https://web.archive.org/web/20201126145825id_/https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
ZIP_SHA256=b4d66deb492d9b0a353b51879152687ed9313897e8e19320d2dc853d738ed8a7
TELEMANOM_COMMIT=2e6c5b6c3558e7835601519b7bdef37c649bdbdc
CSV_URL="https://raw.githubusercontent.com/khundman/telemanom/$TELEMANOM_COMMIT/labeled_anomalies.csv"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl?datasetVersionNumber=1"

cd "$(dirname "$0")/../data"
rm -rf _nasa nasa
mkdir _nasa
if [ "${NASA_SOURCE:-archive}" = kaggle ]; then
  curl -sfL -o _nasa/data.zip "$KAGGLE_URL"
  (cd _nasa && unzip -q data.zip)
  src=$(dirname "$(find _nasa -type d -name train | head -1)")
else
  curl -sfL -o _nasa/data.zip "$ZIP_URL"
  echo "$ZIP_SHA256  _nasa/data.zip" | shasum -a 256 -c -
  (cd _nasa && unzip -q data.zip 'data/train/*' 'data/test/*')
  src=_nasa/data
fi
mkdir nasa
mv "$src/train" "$src/test" nasa/
curl -sfL -o nasa/labeled_anomalies.csv "$CSV_URL"
rm -rf _nasa

sums=$( (cd nasa && find train test -name '*.npy' | sort && echo labeled_anomalies.csv) \
        | (cd nasa && xargs shasum -a 256))
if [ -f nasa_sha256.txt ]; then
  (cd nasa && shasum -a 256 -c --quiet ../nasa_sha256.txt)
  echo "checksums match data/nasa_sha256.txt"
else
  echo "$sums" > nasa_sha256.txt
fi
echo "NASA SMAP/MSL ready at data/nasa"

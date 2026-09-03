#!/usr/bin/env bash
set -euo pipefail

readonly RELEASE_URL="https://digidata.rigsarkivet.dk/download/14001"
readonly PROJECT_ROOT="${1:-${PAPER1_PROJECT_ROOT:-$(pwd)}}"
readonly RAW_DIR="${PROJECT_ROOT}/data/public_genealogy/link_lives/raw/release_2"
readonly ARCHIVE="${RAW_DIR}/14001.zip"
readonly INVENTORY="${RAW_DIR}/14001.inventory.txt"
readonly CHECKSUM="${RAW_DIR}/14001.zip.sha256"

mkdir -p "${RAW_DIR}"

curl \
  --fail \
  --http1.1 \
  --location \
  --connect-timeout 30 \
  --retry 20 \
  --retry-all-errors \
  --retry-delay 10 \
  --continue-at - \
  "${RELEASE_URL}" \
  --output "${ARCHIVE}"

unzip -t "${ARCHIVE}"
unzip -l "${ARCHIVE}" > "${INVENTORY}"
(
  cd "${RAW_DIR}"
  shasum -a 256 "14001.zip" > "14001.zip.sha256"
)

if [[ -n "${PAPER1_S3_ROOT:-}" ]]; then
  : "${AWS_PROFILE:?Set AWS_PROFILE when PAPER1_S3_ROOT is enabled}"
  readonly S3_PREFIX="${PAPER1_S3_ROOT%/}/data/public_genealogy/link_lives/raw/release_2"
  aws s3 cp "${ARCHIVE}" "${S3_PREFIX}/14001.zip" --profile "${AWS_PROFILE}" --no-progress
  aws s3 cp "${INVENTORY}" "${S3_PREFIX}/14001.inventory.txt" --profile "${AWS_PROFILE}" --no-progress
  aws s3 cp "${CHECKSUM}" "${S3_PREFIX}/14001.zip.sha256" --profile "${AWS_PROFILE}" --no-progress
fi

printf 'LINK_LIVES_RELEASE2_READY\n'

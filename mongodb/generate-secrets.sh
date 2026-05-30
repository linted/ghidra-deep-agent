#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/keyfile" || -f "$SCRIPT_DIR/pwfile" || -f "$SCRIPT_DIR/firmware_pwfile" ]]; then
  read -r -p "keyfile, pwfile, or firmware_pwfile already exists. Overwrite? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# MongoDB keyfile: base64-encoded random bytes (6–1024 chars required)
openssl rand -base64 756 > "$SCRIPT_DIR/keyfile"
chmod 400 "$SCRIPT_DIR/keyfile"
echo "Generated keyfile"

# pwfile: UUID used as the mongotUser password
uuidgen | tr '[:upper:]' '[:lower:]' > "$SCRIPT_DIR/pwfile"
chmod 400 "$SCRIPT_DIR/pwfile"
echo "Generated pwfile (mongotUser password)"

# firmware_pwfile: UUID used as the firmware_user password
uuidgen | tr '[:upper:]' '[:lower:]' > "$SCRIPT_DIR/firmware_pwfile"
chmod 400 "$SCRIPT_DIR/firmware_pwfile"
echo "Generated firmware_pwfile (firmware_user password): $(cat "$SCRIPT_DIR/firmware_pwfile")"


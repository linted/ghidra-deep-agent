#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/keyfile" || -f "$SCRIPT_DIR/pwfile" || -f "$SCRIPT_DIR/firmware_pwfile" || -f "$SCRIPT_DIR/admin_pwfile" ]]; then
  read -r -p "keyfile, pwfile, firmware_pwfile, or admin_pwfile already exists. Overwrite? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# MongoDB keyfile: base64-encoded random bytes (6–1024 chars required)
openssl rand -base64 756 > "$SCRIPT_DIR/keyfile"
chmod 400 "$SCRIPT_DIR/keyfile"
echo "Generated keyfile"

# pwfile: UUID used as the mongotUser password
uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '\n' > "$SCRIPT_DIR/pwfile"
chmod 400 "$SCRIPT_DIR/pwfile"
echo "Generated pwfile (mongotUser password)"

# firmware_pwfile: UUID used as the firmware_user password
uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '\n' > "$SCRIPT_DIR/firmware_pwfile"
chmod 400 "$SCRIPT_DIR/firmware_pwfile"
echo "Generated firmware_pwfile (firmware_user password): $(cat "$SCRIPT_DIR/firmware_pwfile")"

# admin_pwfile: UUID used as the MongoDB admin password
uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '\n' > "$SCRIPT_DIR/admin_pwfile"
chmod 400 "$SCRIPT_DIR/admin_pwfile"
echo "Generated admin_pwfile (admin password): $(cat "$SCRIPT_DIR/admin_pwfile")"


#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/keyfile" || -f "$SCRIPT_DIR/pwfile" ]]; then
  read -r -p "keyfile or pwfile already exists. Overwrite? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# MongoDB keyfile: base64-encoded random bytes (6–1024 chars required)
openssl rand -base64 756 > "$SCRIPT_DIR/keyfile"
chmod 400 "$SCRIPT_DIR/keyfile"
echo "Generated keyfile"

# pwfile: UUID used as the mongotUser password
uuidgen | tr '[:upper:]' '[:lower:]' > "$SCRIPT_DIR/pwfile"
echo "Generated pwfile: $(cat "$SCRIPT_DIR/pwfile")"

echo ""
echo "Update the mongotUser password in init-mongo.sh to match pwfile."

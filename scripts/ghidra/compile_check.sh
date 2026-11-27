#!/usr/bin/env bash
# Compile every Ghidra script in scripts/ghidra/ against a local Ghidra install's
# jars. Catches Java errors and Ghidra API drift before a script is deployed,
# since Ghidra only reports a broken script when it runs.
#
# Needs a JDK matching the Ghidra version (Ghidra 12.2 ships Java 25 class files,
# so JDK 25; set JAVAC to override) and a Ghidra install (set GHIDRA_HOME to the
# directory that contains Ghidra/). Defaults to the Homebrew Ghidra 12.2 install.
set -euo pipefail

cd "$(dirname "$0")"

GHIDRA_HOME="${GHIDRA_HOME:-/opt/homebrew/Cellar/ghidra/12.2/libexec}"
JAVAC="${JAVAC:-/opt/homebrew/opt/openjdk@25/bin/javac}"

if [[ ! -x "$JAVAC" ]]; then
    echo "javac not found at $JAVAC (set JAVAC=...)" >&2
    exit 1
fi
if [[ ! -d "$GHIDRA_HOME/Ghidra" ]]; then
    echo "no Ghidra install at $GHIDRA_HOME (set GHIDRA_HOME=...)" >&2
    exit 1
fi

classpath=$(find "$GHIDRA_HOME/Ghidra" -name '*.jar' -path '*/lib/*' | paste -sd: -)
out=$(mktemp -d "${TMPDIR:-/tmp}/ghidra-scripts-compile.XXXXXX")
trap 'rm -rf "$out"' EXIT

"$JAVAC" -encoding UTF-8 -d "$out" -cp "$classpath" ./*.java
echo "compiled $(ls ./*.java | wc -l | tr -d ' ') script(s) OK"

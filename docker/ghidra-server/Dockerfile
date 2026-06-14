# Ghidra Server (shared repository / "server mode") for the deep-agent stack.
# Downloads a published Ghidra release at build time — no Gradle build needed.
FROM eclipse-temurin:21-jdk

# Keep GHIDRA_VERSION in sync with the ghidra-mcp image. GHIDRA_DATE must match
# the asset filename of the chosen version on the NSA/ghidra GitHub release.
ARG GHIDRA_VERSION=12.1
ARG GHIDRA_DATE=20260513

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget -q --no-check-certificate \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_DATE}.zip" \
      -O ghidra.zip || \
    wget -q --no-check-certificate \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC.zip" \
      -O ghidra.zip
RUN unzip -q ghidra.zip && rm ghidra.zip && mv ghidra_* ghidra

ENV GHIDRA_HOME=/opt/ghidra

# Configure server mode by patching the distribution's server.conf (rather than
# shipping a full copy, so the many wrapper settings track the Ghidra version):
#   -a0  local Ghidra password authentication (the only mode the MCP's
#        GhidraServerManager supports — it authenticates with user+password)
#   -e0  no password expiration, so the service-account password the entrypoint
#        sets does not lapse (default is 1 day)
#   repositories stored under /repositories (a dedicated volume)
# NOTE on remote access: by default the server auto-detects its advertised RMI
# hostname/IP, which is reachable by sibling containers on the shared network.
# If RMI callbacks or the self-signed cert prove problematic, add `-ip <name>`
# and `-ipAlt <ip>` parameters here (see server.conf comments).
RUN sed -i \
      -e 's|^ghidra.repositories.dir=.*|ghidra.repositories.dir=/repositories|' \
      -e 's|^wrapper.app.parameter.1=.*|wrapper.app.parameter.1=-a0|' \
      -e 's|^wrapper.app.parameter.2=.*|wrapper.app.parameter.2=-e0|' \
      "${GHIDRA_HOME}/server/server.conf" \
    && printf 'wrapper.app.parameter.3=${ghidra.repositories.dir}\n' >> "${GHIDRA_HOME}/server/server.conf" \
    && mkdir -p /repositories

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# RMI registry (base), block stream, and admin ports.
EXPOSE 13100 13101 13102
VOLUME ["/repositories"]
ENTRYPOINT ["/entrypoint.sh"]

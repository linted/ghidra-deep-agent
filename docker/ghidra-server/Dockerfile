# Ghidra Server (shared repository / "server mode") for the deep-agent stack.
# Downloads a published Ghidra release at build time — no Gradle build needed.
FROM eclipse-temurin:21-jdk

# Keep GHIDRA_VERSION in sync with the ghidra-mcp image. GHIDRA_DATE must match
# the asset filename of the chosen version on the NSA/ghidra GitHub release.
ARG GHIDRA_VERSION=12.1
ARG GHIDRA_DATE=20260513
# Hostname remote clients use to reach this server. MUST match the compose
# service name (and the ghidra-mcp engine's GHIDRA_SERVER_HOST), because the
# server bakes it into the self-signed cert's Subject Alternative Names. If it
# does not match, the engine's TLS handshake fails with "No subject alternative
# DNS name matching <host> found".
ARG GHIDRA_SERVER_HOSTNAME=ghidra-server

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
#   -ip <hostname>  advertised RMI hostname + self-signed cert SAN. Without it
#        the server only puts its container IP/hostname/localhost in the cert,
#        and a sibling container connecting by service name fails TLS hostname
#        verification. The repository path must remain the LAST positional arg.
#   repositories stored under /repositories (a dedicated volume)
RUN sed -i \
      -e 's|^ghidra.repositories.dir=.*|ghidra.repositories.dir=/repositories|' \
      -e 's|^wrapper.app.parameter.1=.*|wrapper.app.parameter.1=-a0|' \
      -e 's|^wrapper.app.parameter.2=.*|wrapper.app.parameter.2=-e0|' \
      "${GHIDRA_HOME}/server/server.conf" \
    && { \
         printf 'wrapper.app.parameter.3=-ip\n'; \
         printf 'wrapper.app.parameter.4=%s\n' "${GHIDRA_SERVER_HOSTNAME}"; \
         printf 'wrapper.app.parameter.5=${ghidra.repositories.dir}\n'; \
       } >> "${GHIDRA_HOME}/server/server.conf" \
    && mkdir -p /repositories

# Compile the headless repository-seeding helper (see CreateRepository.java).
# svrAdmin cannot create repositories, so first-boot provisioning of a default
# shared repo goes through the Ghidra RMI client API instead. We compile against
# the Framework jars (plus Features/Base for the headless app configuration) and
# persist the classpath so the entrypoint can run the helper without rebuilding
# it. GHIDRA_HOME is fixed, so the jar set is stable across runs.
COPY CreateRepository.java /opt/repo-tools/CreateRepository.java
RUN set -eu; \
    CP=""; \
    for jar in "${GHIDRA_HOME}"/Ghidra/Framework/*/lib/*.jar \
               "${GHIDRA_HOME}"/Ghidra/Features/Base/lib/*.jar; do \
      [ -f "${jar}" ] && CP="${CP}:${jar}"; \
    done; \
    CP="${CP#:}"; \
    javac -cp "${CP}" -d /opt/repo-tools /opt/repo-tools/CreateRepository.java; \
    printf '%s' "${CP}" > /opt/repo-tools/classpath

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# RMI registry (base), block stream, and admin ports.
EXPOSE 13100 13101 13102
VOLUME ["/repositories"]
ENTRYPOINT ["/entrypoint.sh"]

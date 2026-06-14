# Combined headless Ghidra engine + Python MCP bridge in a single container.
#
#   web (MultiServerMCPClient) --MCP--> bridge :8081 /mcp --REST--> engine :8089
#
# The Java headless server speaks the GhidraMCP REST API (not MCP); the Python
# bridge translates MCP <-> REST. Both run here so the agent gets one MCP
# endpoint per analysis engine. Artifacts are downloaded from published GitHub
# releases at build time — no Maven/Gradle build required.
FROM eclipse-temurin:21-jdk

# Keep GHIDRA_VERSION aligned with the ghidra-server image and with the Ghidra
# version the GhidraMCP release targets. GHIDRA_DATE must match the NSA/ghidra
# release asset filename.
ARG GHIDRA_VERSION=12.1
ARG GHIDRA_DATE=20260513
ARG GHIDRA_MCP_VERSION=5.13.1

# JDK (not JRE) is required: Ghidra compiles GhidraScript .java files on the fly
# via javax.tools.ToolProvider.getSystemJavaCompiler(), which is null on a JRE.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip curl ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Ghidra distribution ---
WORKDIR /opt
RUN wget -q --no-check-certificate \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_DATE}.zip" \
      -O ghidra.zip || \
    wget -q --no-check-certificate \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC.zip" \
      -O ghidra.zip
RUN unzip -q ghidra.zip && rm ghidra.zip && mv ghidra_* ghidra
ENV GHIDRA_HOME=/opt/ghidra

# --- GhidraMCP headless jar + Python bridge (from the GitHub release) ---
# The release's GhidraMCP-<ver>.zip is a Ghidra extension whose
# GhidraMCP/lib/GhidraMCP-<ver>.jar bundles com.xebyte.headless.* (the headless
# server). The bridge and its deps (mcp, requests) ship as separate assets.
WORKDIR /app
RUN set -eux; \
    BASE="https://github.com/bethington/ghidra-mcp/releases/download/v${GHIDRA_MCP_VERSION}"; \
    wget -q "${BASE}/GhidraMCP-${GHIDRA_MCP_VERSION}.zip" -O ghidramcp.zip; \
    unzip -q ghidramcp.zip -d ghidramcp; \
    cp "ghidramcp/GhidraMCP/lib/GhidraMCP-${GHIDRA_MCP_VERSION}.jar" /app/GhidraMCP.jar; \
    rm -rf ghidramcp ghidramcp.zip; \
    wget -q "${BASE}/bridge_mcp_ghidra.py" -O /app/bridge_mcp_ghidra.py; \
    wget -q "${BASE}/requirements.txt" -O /app/requirements.txt; \
    pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

ENV GHIDRA_MCP_PORT=8089
ENV MCP_BRIDGE_PORT=8081
ENV JAVA_OPTS="-Xmx4g -XX:+UseG1GC"

ENV IMPORTER_PORT=8082

RUN mkdir -p /data /projects
# Binary import service (stdlib only): runs analyzeHeadless to commit uploaded
# binaries into the shared repo, since the web container has no Ghidra.
COPY importer.py /app/importer.py
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 8089 stays internal (engine REST); 8081 is the MCP endpoint the agent uses;
# 8082 is the import service (internal — reached by the web service by name).
EXPOSE 8089 8081 8082
VOLUME ["/data", "/projects"]

# Health = engine REST API up. Generous start period: Ghidra init is slow.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS "http://localhost:${GHIDRA_MCP_PORT}/check_connection" >/dev/null || exit 1

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]

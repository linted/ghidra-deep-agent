# Combined headless Ghidra engine + Python MCP bridge in a single container.
#
#   web (MultiServerMCPClient) --MCP--> bridge :8081 /mcp --REST--> engine :8089
#
# The Java headless server speaks the GhidraMCP REST API (not MCP); the Python
# bridge translates MCP <-> REST. Both run here so the agent gets one MCP
# endpoint per analysis engine.
#
# Unlike the upstream image, we build the GhidraMCP headless jar FROM SOURCE in
# a Maven builder stage (see GHIDRA_MCP_REPO/REF below) rather than downloading a
# published GitHub release. This lets us track a fork that carries unreleased
# changes (e.g. open_program_from_server). The bridge_mcp_ghidra.py + its
# requirements.txt are taken from that same source tree.

# =============================================================================
# Stage 1: Builder — download Ghidra, install its jars to the Maven local repo,
# clone the GhidraMCP source, and build the headless jar.
# =============================================================================
FROM eclipse-temurin:21-jdk AS builder

# Keep GHIDRA_VERSION aligned with the ghidra-server image and with pom.xml's
# <ghidra.version> in the source we build. The Maven build resolves ghidra:*
# at this exact version, and the install-file steps stamp it onto the jars we
# extract from the Ghidra release zip — a mismatch breaks dependency resolution.
# GHIDRA_DATE must match the NSA/ghidra release asset filename.
ARG GHIDRA_VERSION=12.1.2
ARG GHIDRA_DATE=20260605

# Source of the GhidraMCP headless server. Defaults to the fork branch we're
# tracking (carries the headless open + Ghidra-Server-open fixes); override
# GHIDRA_MCP_REF to pin a specific commit for reproducible builds. NOTE: the
# branch name contains a slash, so it must be reachable as refs/heads/<ref>.
ARG GHIDRA_MCP_REPO=https://github.com/linted/ghidra-mcp.git
ARG GHIDRA_MCP_REF=linted/main

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip maven git ca-certificates \
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

# Install the Ghidra jars the GhidraMCP pom depends on into the Maven local
# repo (Ghidra isn't published to Maven Central). This list mirrors the fork's
# own docker/Dockerfile — keep it in sync if the pom's <dependencies> change.
RUN mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Generic/lib/Generic.jar \
        -DgroupId=ghidra -DartifactId=Generic -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/SoftwareModeling/lib/SoftwareModeling.jar \
        -DgroupId=ghidra -DartifactId=SoftwareModeling -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Project/lib/Project.jar \
        -DgroupId=ghidra -DartifactId=Project -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Docking/lib/Docking.jar \
        -DgroupId=ghidra -DartifactId=Docking -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Utility/lib/Utility.jar \
        -DgroupId=ghidra -DartifactId=Utility -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Gui/lib/Gui.jar \
        -DgroupId=ghidra -DartifactId=Gui -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/FileSystem/lib/FileSystem.jar \
        -DgroupId=ghidra -DartifactId=FileSystem -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Help/lib/Help.jar \
        -DgroupId=ghidra -DartifactId=Help -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Features/Base/lib/Base.jar \
        -DgroupId=ghidra -DartifactId=Base -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Features/Decompiler/lib/Decompiler.jar \
        -DgroupId=ghidra -DartifactId=Decompiler -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/DB/lib/DB.jar \
        -DgroupId=ghidra -DartifactId=DB -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Framework/Emulation/lib/Emulation.jar \
        -DgroupId=ghidra -DartifactId=Emulation -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Debug/Debugger-api/lib/Debugger-api.jar \
        -DgroupId=ghidra -DartifactId=Debugger-api -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Debug/Framework-TraceModeling/lib/Framework-TraceModeling.jar \
        -DgroupId=ghidra -DartifactId=Framework-TraceModeling -Dversion=${GHIDRA_VERSION} -Dpackaging=jar && \
    mvn -q install:install-file -Dfile=/opt/ghidra/Ghidra/Debug/Debugger-rmi-trace/lib/Debugger-rmi-trace.jar \
        -DgroupId=ghidra -DartifactId=Debugger-rmi-trace -Dversion=${GHIDRA_VERSION} -Dpackaging=jar

# --- Clone GhidraMCP source and build the headless jar ---
# The `headless` profile produces a single fat jar (target/GhidraMCP-*.jar) that
# bundles runtime deps but excludes Ghidra's own jars (provided at runtime via
# the classpath the entrypoint builds). Tests are skipped for the image build.
WORKDIR /build
RUN git clone "${GHIDRA_MCP_REPO}" . && git checkout "${GHIDRA_MCP_REF}"
RUN mvn clean package -P headless -DskipTests -q

# =============================================================================
# Stage 2: Runtime — Ghidra + the built headless jar + the Python MCP bridge.
# =============================================================================
# JDK (not JRE) is required: Ghidra compiles GhidraScript .java files on the fly
# via javax.tools.ToolProvider.getSystemJavaCompiler(), which is null on a JRE.
FROM eclipse-temurin:21-jdk

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ghidra (already downloaded/unpacked in the builder).
COPY --from=builder /opt/ghidra /opt/ghidra
ENV GHIDRA_HOME=/opt/ghidra

# The built headless jar plus the Python MCP bridge, all from the same source
# tree we compiled. The bridge is now a hatchling package under python/ (it
# declares only `mcp` as a runtime dep — it talks to the engine over stdlib
# http.client), installed via PEP 517 from its pyproject. README is required by
# the build (pyproject `readme = "README.md"`).
WORKDIR /app
COPY --from=builder /build/target/GhidraMCP-*.jar /app/GhidraMCP.jar
COPY --from=builder /build/pyproject.toml /build/README.md /app/bridge-src/
COPY --from=builder /build/python /app/bridge-src/python
RUN pip install --no-cache-dir --break-system-packages /app/bridge-src

ENV GHIDRA_MCP_PORT=8089
ENV MCP_BRIDGE_PORT=8081
ENV JAVA_OPTS="-Xmx4g -XX:+UseG1GC"

RUN mkdir -p /data /projects
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 8089 stays internal (engine REST); 8081 is the MCP endpoint the agent uses.
# (Binary import now lives in the ghidra-server container — see its importer.py.)
EXPOSE 8089 8081
VOLUME ["/data", "/projects"]

# Health = engine REST API up. Generous start period: Ghidra init is slow.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS "http://localhost:${GHIDRA_MCP_PORT}/check_connection" >/dev/null || exit 1

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]

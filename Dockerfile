FROM node:22-bookworm-slim

# python3 for server.py; ca-certificates for the claude CLI's own HTTPS calls
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Official Claude Code CLI (Linux build), installed and allowed to run its postinstall
# (downloads/builds the native claude binary -- same requirement as the host install).
RUN npm install -g --allow-scripts=@anthropic-ai/claude-code @anthropic-ai/claude-code

WORKDIR /app
COPY server.py ./
COPY vendor/ ./vendor/

ENV CLAUDE_SUBSCRIPTION_DIRECTSDK_COMMAND=claude
ENV EISY_SHIM_HOST=0.0.0.0
ENV EISY_SHIM_PORT=8799
# Claude Code's login directory -- mount a named volume here so `claude auth login`
# survives container rebuilds/restarts.
ENV CLAUDE_CONFIG_DIR=/root/.claude

EXPOSE 8799

CMD ["python3", "server.py"]

# What If studio — the chat-driven AI diffusion animation app.
# Multi-stage: the build stage type-checks and compiles the TypeScript, the runtime stays lean.

FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    WHATIF_DATA=/data \
    WHATIF_HOST=0.0.0.0 \
    WHATIF_PORT=3020

# python3 + pillow/numpy for image sheets/refs, ffmpeg for I2V assembly & -14 LUFS mastering, git for checkpoints.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg git ca-certificates curl util-linux \
      fonts-liberation fonts-dejavu-core \
    && python3 -m pip install --no-cache-dir --break-system-packages pillow numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY pipeline ./pipeline
COPY web ./web
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /data

# No USER here: the entrypoint fixes /data ownership as root, then drops to uid 1000 (node) with setpriv.
EXPOSE 3020
ENTRYPOINT ["entrypoint.sh"]

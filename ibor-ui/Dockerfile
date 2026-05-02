# ─────────────────────────────────────────────────────────────────────────────
# IBOR UI (React + Vite) Dockerfile
# Multi-stage build: Build → Nginx server
# Backend URLs are injected at runtime via GATEWAY_URL / MIDDLEWARE_URL env vars.
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build with Node.js
FROM node:22-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Serve with Nginx (runtime-configurable backend URLs)
FROM nginx:alpine

# Install gettext for envsubst
RUN apk add --no-cache gettext

# Copy built artifacts from builder
COPY --from=builder /build/dist /usr/share/nginx/html

# Write main nginx.conf (non-root compatible)
RUN cat > /etc/nginx/nginx.conf <<'EOF'
pid /tmp/nginx.pid;
worker_processes auto;
events { worker_connections 1024; }
http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;
    include /etc/nginx/conf.d/*.conf;
}
EOF

# Write Nginx site config TEMPLATE
# ${GATEWAY_URL}, ${MIDDLEWARE_URL}, ${API_KEY}, ${PORT}, ${RESOLVER} substituted at runtime.
# All other nginx $variables are single-dollar and left untouched by envsubst.
RUN cat > /etc/nginx/conf.d/default.conf.template <<'EOF'
server {
    listen ${PORT};
    server_name _;

    # DNS resolver read from /etc/resolv.conf at startup (resolves .railway.internal)
    resolver ${RESOLVER} valid=30s;

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /analyst/ {
        # Private network — variable forces per-request DNS via resolver above
        set $gateway_url ${GATEWAY_URL};
        proxy_pass $gateway_url;
        include /etc/nginx/conf.d/gateway_auth.conf;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    location /api/ {
        # Variable forces per-request DNS lookup via resolver above.
        # proxy_ssl_server_name on: sends SNI when MIDDLEWARE_URL is https://*.
        set $middleware_url ${MIDDLEWARE_URL};
        proxy_pass $middleware_url;
        proxy_ssl_server_name on;
        proxy_set_header Host $proxy_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
        proxy_connect_timeout 10s;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }

    gzip on;
    gzip_types text/plain text/css text/javascript application/javascript application/json;
    gzip_min_length 1000;
}
EOF

# Entrypoint: substitute env vars into nginx config, then start nginx
RUN cat > /docker-entrypoint-nginx.sh <<'SCRIPT'
#!/bin/sh
set -e
export PORT="${PORT:-5173}"
export GATEWAY_URL="${GATEWAY_URL:-http://ibor-ai-gateway.railway.internal:8000}"
export MIDDLEWARE_URL="${MIDDLEWARE_URL:-https://ibor-middleware-production.up.railway.app}"
export API_KEY="${API_KEY:-}"
# Write API key as a standalone nginx include file — avoids envsubst edge cases
printf 'proxy_set_header X-API-Key "%s";\n' "$API_KEY" > /etc/nginx/conf.d/gateway_auth.conf
# Substitute remaining env vars
RESOLVER=$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf)
export RESOLVER="${RESOLVER:-8.8.8.8}"
export GATEWAY_URL="${GATEWAY_URL:-http://ibor-ai-gateway.railway.internal:8000}"
export MIDDLEWARE_URL="${MIDDLEWARE_URL:-https://ibor-middleware-production.up.railway.app}"
export PORT="${PORT:-5173}"
envsubst '${GATEWAY_URL} ${MIDDLEWARE_URL} ${PORT} ${RESOLVER}' \
  < /etc/nginx/conf.d/default.conf.template \
  > /etc/nginx/conf.d/default.conf
exec nginx -g "daemon off;"
SCRIPT
RUN chmod +x /docker-entrypoint-nginx.sh

# Create non-root user
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser && \
    chown -R appuser:appuser /usr/share/nginx/html /var/cache/nginx /var/log/nginx /etc/nginx /tmp

USER appuser

EXPOSE 5173

# Railway overrides PORT at runtime; local Docker keeps 5173

CMD ["/docker-entrypoint-nginx.sh"]

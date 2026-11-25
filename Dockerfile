# Build stage
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags '-s -w' -o /out/mlai-bridge .

# Runtime: needs Node + the mlai CLI, which the container signs in to at runtime.
FROM node:22-alpine
ARG MLAI_CODE_VERSION=0.3.27
RUN npm install -g mlai-code@${MLAI_CODE_VERSION} && apk add --no-cache ca-certificates tini
COPY --from=build /out/mlai-bridge /usr/local/bin/mlai-bridge
RUN mkdir -p /data/workspace /home/node/.mlai && chown -R node:node /data /home/node/.mlai
USER node
ENV HOME=/home/node
WORKDIR /data
# 0.0.0.0 is needed to publish the port. The bridge refuses to start on a
# non-loopback address without MLAI_API_KEY; pass it at run time (-e / --env-file), never bake it in.
ENV MLAI_BIND=0.0.0.0 \
    MLAI_WORKDIR=/data/workspace \
    MLAI_MODEL=pixis-opus \
    MLAI_PORT=8197
EXPOSE 8197
# Mount a signed-in ~/.mlai into the container (read-write: the CLI refreshes tokens).
VOLUME ["/home/node/.mlai"]
ENTRYPOINT ["/sbin/tini", "--", "mlai-bridge"]
CMD ["-port", "8197"]

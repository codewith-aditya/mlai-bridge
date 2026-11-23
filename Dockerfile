# Build stage
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags '-s -w' -o /out/mlai-bridge .

# Runtime: needs Node + the mlai CLI, which the container signs in to at runtime.
FROM node:22-alpine
RUN npm install -g mlai-code && apk add --no-cache ca-certificates
COPY --from=build /out/mlai-bridge /usr/local/bin/mlai-bridge
WORKDIR /data
ENV MLAI_WORKDIR=/data/workspace \
    MLAI_MODEL=pixis-opus \
    MLAI_PORT=8197
EXPOSE 8197
# Mount a signed-in ~/.mlai into the container (read-write: the CLI refreshes tokens).
VOLUME ["/root/.mlai"]
ENTRYPOINT ["mlai-bridge"]
CMD ["-port", "8197"]

BINARY := mlai-bridge

.PHONY: build run vet fmt test install clean docker

build:
	CGO_ENABLED=0 go build -trimpath -ldflags '-s -w' -o $(BINARY) .

run: build
	./$(BINARY) -port 8197

vet:
	go vet ./...

fmt:
	gofmt -w *.go

test:
	go test ./...

install: build
	install -m 0755 $(BINARY) /usr/local/bin/$(BINARY)
	ln -sf $(BINARY) /usr/local/bin/ml-wrap

docker:
	docker build -t mlai-bridge:latest .

clean:
	rm -f $(BINARY)

# smoke test against a running bridge
smoke:
	curl -sS http://127.0.0.1:8197/healthz
	curl -sS http://127.0.0.1:8197/v1/chat/completions -H 'Content-Type: application/json' \
		-d '{"model":"pixis-opus","messages":[{"role":"user","content":"say ok"}]}'

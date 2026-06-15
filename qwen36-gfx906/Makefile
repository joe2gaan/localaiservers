.PHONY: help build run down logs status clean dist-clean check-space check-gpu

SHELL := /bin/sh
BUNDLE_COMPOSE := docker-compose.deploy.yml
BUNDLE_COMPOSE_ENV := deploy.env
RUNTIME_ENV := .deploy.runtime.env
LOCAL_DOCKER_TAR := $(wildcard .*.image.tar)
LOCAL_DOCKER_TAR_FALLBACK := .*.image.tar
DEPLOY_IMAGE := $(shell if [ -f "$(BUNDLE_COMPOSE_ENV)" ]; then awk -F"'" 'BEGIN{img="qwen36-gfx906-c1-topk8-fastpath-reproducible"} /^DEPLOY_IMAGE=/{img=$$2; exit} END{print img}' "$(BUNDLE_COMPOSE_ENV)"; else printf '%s\n' qwen36-gfx906-c1-topk8-fastpath-reproducible; fi)
TEMP_DOCKER_ROOT := $(shell if [ -f "$(BUNDLE_COMPOSE_ENV)" ]; then awk -F"'" '/^DOCKER_TEMP_DATA_ROOT_DIR=/ {print $$2; exit}' "$(BUNDLE_COMPOSE_ENV)"; fi)
CONTAINER_NAME := $(shell if [ -f "$(BUNDLE_COMPOSE_ENV)" ]; then awk -F"'" '/^CONTAINER_NAME=/ {print $$2; exit}' "$(BUNDLE_COMPOSE_ENV)"; fi)

help:
	@echo "Targets:"
	@echo "  build              - build image using docker compose and current env file."
	@echo "  run                - full deploy workflow (calls ./deploy.sh)."
	@echo "  run-temp-root      - build with temporary Docker data-root, then restore daemon config and start."
	@echo "  down               - stop and remove compose service."
	@echo "  logs               - tail service logs."
	@echo "  status             - show running service status."
	@echo "  clean              - remove generated local artifacts (logs/env backups/tar/temp-root)."
	@echo "  dist-clean         - removes local caches and mounted runtime/model cache dirs."
	@echo "  check-space        - print docker root and disk usage report."
	@echo "  check-gpu          - run GPU VRAM preflight check only."

build:
	@if [ -f "$(RUNTIME_ENV)" ]; then \
	  docker compose --env-file "$(RUNTIME_ENV)" -f "$(BUNDLE_COMPOSE)" build; \
	else \
	  docker compose --env-file "$(BUNDLE_COMPOSE_ENV)" -f "$(BUNDLE_COMPOSE)" build; \
	fi

run:
	./deploy.sh

run-temp-root:
	DOCKER_TEMP_DATA_ROOT_ENABLED=1 \
	DOCKER_TEMP_DATA_ROOT_DIR="$(if $(strip $(TEMP_DOCKER_ROOT)),$(TEMP_DOCKER_ROOT),./.docker-data-root)" \
	./deploy.sh

down:
	@if [ -f "$(RUNTIME_ENV)" ]; then \
	  docker compose --env-file "$(RUNTIME_ENV)" -f "$(BUNDLE_COMPOSE)" down --remove-orphans; \
	else \
	  docker compose --env-file "$(BUNDLE_COMPOSE_ENV)" -f "$(BUNDLE_COMPOSE)" down --remove-orphans; \
	fi

logs:
	docker compose -f "$(BUNDLE_COMPOSE)" logs -f --tail=200

status:
	@if [ -n "$(CONTAINER_NAME)" ]; then \
	  docker ps --filter "name=$(CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
	  docker ps --filter "name=qwen36" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	fi

check-space:
	@docker_root="$$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"; \
	echo "Docker root dir: $${docker_root:-unknown}"; \
	if [ -n "$${docker_root}" ] && [ -d "$${docker_root}" ]; then \
	  if root_usage="$$(du -sh "$${docker_root}" 2>/dev/null | awk '{print $$1}')"; then \
	    if [ -n "$${root_usage}" ]; then echo "space used in $${docker_root}: $${root_usage}"; else echo "warning: docker root exists but usage unavailable"; fi; \
	  else \
	    echo "warning: docker root usage unavailable (permission denied)"; \
	  fi; \
	else \
	  echo "warning: docker root directory unavailable or not mounted in this shell"; \
	fi; \
	for p in "$$(pwd)/hf_cache" "$$(pwd)/runtime" "$$(pwd)/.docker-data-root"; do \
	  if [ -d "$${p}" ]; then \
	    size="$$(du -sh "$${p}" 2>/dev/null | awk '{print $$1}')"; \
	    echo "space used in $${p}: $${size}"; \
	  fi; \
	done; \
	docker system df -v

check-gpu:
	SKIP_DISK_SPACE_CHECK=1 \
	CHECK_GPU_ONLY=1 \
	./deploy.sh

clean:
	@rm -f .deploy.runtime.env .deploy.daemon.json.backup
	@rm -f $(LOCAL_DOCKER_TAR) $(LOCAL_DOCKER_TAR_FALLBACK)
	@rm -rf .docker-data-root "$(if $(strip $(TEMP_DOCKER_ROOT)),$(TEMP_DOCKER_ROOT),)"
	@rm -rf .logs/*.log .logs/*.pid
	@echo "artifact cleanup complete"

dist-clean: clean
	@rm -rf runtime hf_cache
	@echo "runtime and hf cache removed"

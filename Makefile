run:
	docker compose -f compose.yaml up -d --wait --wait-timeout 120
	uv run python -m scripts.init_raw_data

init:
	uv sync
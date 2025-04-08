# For local use
.PHONY: local-run
local-run:
	python3 main.py


# For Docker
.PHONY: db-up
db-up:
	docker compose up -d mongo

.PHONY: db-stop
db-stop:
	docker stop mongo


.PHONY: down
down:
	docker compose down

# Rocotic Dialog Assistant
### Run a service
```bash
# step 1 run proxed services
export COMPOSE_FILE=compose.proxy.yaml
# set -x COMPOSE_FILE compose.proxy.yaml
docker compose up -d --build
# step 2 run your local service
export COMPOSE_FILE=compose.base.yaml:compose.dev.yaml
# set -x COMPOSE_FILE compose.base.yaml:compose.dev.yaml
docker compose up -d --build $SERVICE
```

### Test your service

```bash
export COMPOSE_FILE=compose.base.yaml:compose.dev.yaml
# set -x COMPOSE_FILE compose.base.yaml:compose.dev.yaml
docker compose up -d --build $SERVICE
docker compose exec $SERVICE bash -c "poetry run pytest -vv"
# or with stdout `docker compose exec $SERVICE bash -c "poetry run pytest -vv  --capture=tee-sys"`
```



#### Bare requests
To a service by `curl`
```bash
curl --location --request POST 'http://service:8000/invoke' \
    --header 'Content-Type: application/json' \
    --data-raw '{}'
```

To assistant by `curl`
```bash
curl -X POST http://chatsky.ipavlov.mipt.ru:8000 \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "asdasd",
        "payload": {"type": "text_request", "text_request":"Привет Квант!"},
        "telemetry": {}
    }'
```
### Contribute

Before you start contributing, use [Ruff](https://github.com/charliermarsh/ruff) to check your code style  and update it.
```bash
ruff format .
ruff check .
```
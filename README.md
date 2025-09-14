# csce439-ember (Dockerized Defender)

## Build
docker build -f defender/Dockerfile -t csce439-ember .

## Run
docker run --rm -p 8080:8080 --name av-defender csce439-ember

## Test
curl -s http://127.0.0.1:8080/model | jq
curl -s -X POST -H "Content-Type: application/octet-stream" \
  --data-binary @/path/to/sample.exe \
  http://127.0.0.1:8080 | jq

## Debug (optional)
curl -s -X POST -H "Content-Type: application/octet-stream" \
  --data-binary @/path/to/sample.exe \
  "http://127.0.0.1:8080?debug=1" | jq

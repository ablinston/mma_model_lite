# Have hard-coded a token here for testing. This is not the production auth token
curl -i -X POST localhost:8000/predict \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer kLZQkdejqSonY-ggJjzlpRzZr7yKtXCbsdqg12hzrrs" \
    -d @/workspace/tests/serving/test.json
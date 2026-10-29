# # Have hard-coded a token here for testing. This is not the production auth token

# curl -i -X POST localhost:8000/predict \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer kLZQkdejqSonY-ggJjzlpRzZr7yKtXCbsdqg12hzrrs" \
#     -d @/workspace/tests/serving/test.json

# Test the service running in Kubernetes on Pi
curl -i -X POST mma-model.192.168.68.51.nip.io:8081/predict \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer kLZQkdejqSonY-ggJjzlpRzZr7yKtXCbsdqg12hzrrs" \
    -d @/workspace/tests/serving/test.json
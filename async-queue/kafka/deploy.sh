set -euo pipefail

NAMESPACE="infergrid"
RELEASE="kafka"
CHART_VERSION="32.4.3"
TOPICS=("infergrid.model-a.requests" "infergrid.model-b.requests")

echo "==> Adding Bitnami Helm repo"
helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null || true
helm repo update

echo "==> Installing/upgrading Kafka"
helm upgrade --install "$RELEASE" bitnami/kafka \
  --version "$CHART_VERSION" \
  --namespace "$NAMESPACE" \
  --values async-queue/kafka/values.yaml \
  --wait \
  --timeout 5m

echo "==> Waiting for Kafka broker to be ready"
kubectl rollout status statefulset/kafka-controller \
  --namespace "$NAMESPACE" \
  --timeout 120s

echo "==> Validating topics"
for topic in "${TOPICS[@]}"; do
  echo -n "    $topic ... "
  kubectl exec -n "$NAMESPACE" kafka-controller-0 -- \
    kafka-topics.sh \
      --bootstrap-server localhost:9092 \
      --describe \
      --topic "$topic" \
    | grep -F "Topic: $topic" >/dev/null && echo "OK" || echo "MISSING"
done

echo ""
echo "   Kafka deployed."
echo ""
echo "Broker address for services:"
echo "  kafka.infergrid.svc.cluster.local:9092"
echo ""
echo "To produce a test message:"
echo "  kubectl exec -n infergrid -it kafka-controller-0 -- \\"
echo "    kafka-console-producer.sh \\"
echo "      --bootstrap-server localhost:9092 \\"
echo "      --topic infergrid.model-a.requests"
echo ""
echo "To consume test messages:"
echo "  kubectl exec -n infergrid -it kafka-controller-0 -- \\"
echo "    kafka-console-consumer.sh \\"
echo "      --bootstrap-server localhost:9092 \\"
echo "      --topic infergrid.model-a.requests \\"
echo "      --from-beginning"

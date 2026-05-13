set -euo pipefail
 
CLUSTER_NAME="infergrid"
 
echo "==> Deleting kind cluster: $CLUSTER_NAME"
kind delete cluster --name "$CLUSTER_NAME"
echo "Cluster deleted."
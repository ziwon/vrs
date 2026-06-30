# k3s GPU Validation

This guide validates the production-shaped GPU path on a single-node k3s host.
It is intentionally narrower than a cloud reference architecture: it checks
that Kubernetes can schedule the native DeepStream worker, run the API and
console, publish detections through the Redis bridge sidecar, and expose the
same portable VRS interfaces.

It does not prove detector accuracy, production throughput, retention policy,
or site-specific alert quality. Treat those as separate acceptance gates.

## Prerequisites

- A k3s node with an NVIDIA GPU.
- NVIDIA driver and container runtime configured for k3s/containerd.
- NVIDIA device plugin or GPU Operator exposing `nvidia.com/gpu`.
- `helm`, `kubectl`, `docker`, and access to this repository on the host.
- DeepStream input media at `/data/vrs/input.mp4`.
- TensorRT/DeepStream model artifacts under `/data/vrs/models`.

The default k3s packaged components are enough for this smoke path when they
are left enabled: CoreDNS, local-path storage, metrics-server, Traefik, and
ServiceLB. VRS does not require Traefik for the baseline validation because the
chart exposes API and console as Kubernetes Services. If your k3s installation
disabled local-path storage or ServiceLB, provide equivalent CSI storage and
service exposure through your platform.

Check the GPU resource before installing VRS:

```bash
nvidia-smi
kubectl describe node "$(kubectl get nodes -o name | head -1 | cut -d/ -f2)" \
  | grep -A5 -E 'Allocatable|nvidia.com/gpu'
```

Expected signal: `nvidia.com/gpu` appears under node capacity/allocatable.

Check the platform defaults as well:

```bash
kubectl get storageclass
kubectl get runtimeclass || true
kubectl get deploy -A | grep -E 'traefik|local-path|metrics-server' || true
```

The same read-only checks are available through Just:

```bash
just k3s-preflight
```

## Build And Import Images

For a local k3s containerd workflow:

```bash
docker build -t vrs:latest -f Dockerfile.backend .
docker build -t vrs-console:latest -f Dockerfile.console .
docker build -t vrs-deepstream:ds8 -f Dockerfile.deepstream .

docker save vrs:latest -o /tmp/vrs-api.tar
docker save vrs-console:latest -o /tmp/vrs-console.tar
docker save vrs-deepstream:ds8 -o /tmp/vrs-deepstream.tar

sudo k3s ctr images import /tmp/vrs-api.tar
sudo k3s ctr images import /tmp/vrs-console.tar
sudo k3s ctr images import /tmp/vrs-deepstream.tar
```

If your k3s node pulls from a registry, push the same images there and override
the chart image repositories through values or `--set`.

## Prepare Host Paths

The `values-k3s-gpu.yaml` profile uses hostPath volumes for the native
DeepStream file-source smoke path and SeaweedFS for S3-compatible evidence
storage:

```bash
sudo mkdir -p /data/vrs/models
sudo cp /path/to/input.mp4 /data/vrs/input.mp4
# Copy TensorRT engine/model artifacts expected by the DeepStream configs.
sudo cp /path/to/models/* /data/vrs/models/
```

The profile mounts:

- `/data/vrs` as `/data/vrs`
- `/data/vrs/models` as `/models`
- an `emptyDir` as `/tmp/vrs` for `vrsmeta` JSONL output and the Redis bridge
- a SeaweedFS PVC for S3-compatible evidence storage

## Render And Install

```bash
helm lint charts/vrs
helm template vrs charts/vrs -f charts/vrs/values-k3s-gpu.yaml

helm install vrs charts/vrs \
  -f charts/vrs/values-k3s-gpu.yaml \
  --set image.repository=vrs \
  --set image.tag=latest \
  --set image.pullPolicy=IfNotPresent
```

If you want a browser-visible console on a single-node k3s host without adding
Gateway API yet, expose only the console service through k3s ServiceLB:

```bash
helm upgrade --install vrs charts/vrs \
  -f charts/vrs/values-k3s-gpu.yaml \
  --set image.repository=vrs \
  --set console.service.type=LoadBalancer
kubectl get svc vrs-vrs-console
```

The API can remain internal because the console nginx configuration proxies
`/api/*` to the in-cluster API service. For locked-down systems, keep both
services as `ClusterIP` and use `kubectl port-forward`.

If your k3s cluster uses an NVIDIA runtime class, pass it explicitly:

```bash
helm upgrade --install vrs charts/vrs \
  -f charts/vrs/values-k3s-gpu.yaml \
  --set image.repository=vrs \
  --set deepstreamWorker.runtimeClassName=nvidia
```

If your GPU nodes carry custom labels or taints, set workload-specific
scheduling fields:

```bash
helm upgrade --install vrs charts/vrs \
  -f charts/vrs/values-k3s-gpu.yaml \
  --set image.repository=vrs \
  --set deepstreamWorker.nodeSelector.vrs\\.ai/gpu-node=true
```

The current profile uses the NVIDIA device-plugin resource name
`nvidia.com/gpu`. DRA, ResourceClaims, MIG-aware placement, GPU Operator
feature discovery labels, and a VRS multi-GPU scheduler are platform/future
extensions, not baseline chart requirements. Keep those concerns in an overlay
until VRS owns multi-GPU scheduling policy directly.

Gateway API follows the same rule. The VRS chart provides stable Services for
API and console. HTTPRoute, GatewayClass, certificates, DNS, Cilium, Traefik,
or cloud load balancer integration should live in a platform overlay. A simple
k3s smoke should prove that the console and API work before adding that layer.

## Validate

Check scheduling and GPU request:

```bash
kubectl get pods -o wide
kubectl describe deploy/vrs-vrs-deepstream-worker | grep -A8 -E 'Limits|Node-Selectors|Tolerations|RuntimeClass'
```

Check worker and bridge logs:

```bash
kubectl logs deploy/vrs-vrs-deepstream-worker -c deepstream-worker
kubectl logs deploy/vrs-vrs-deepstream-worker -c detection-publisher
```

Check API and console:

```bash
kubectl port-forward svc/vrs-vrs-api 8000:8000
curl -fsS http://127.0.0.1:8000/api/health

kubectl port-forward svc/vrs-vrs-console 5173:80
curl -fsS http://127.0.0.1:5173/api/health
```

## Success Criteria

The k3s GPU validation is successful when:

- the DeepStream worker pod is scheduled on the GPU node;
- the worker requests `nvidia.com/gpu: 1`;
- the native DS8 container starts and runs the configured pipeline;
- `vrsmeta` writes `detection.v1` JSONL under `/tmp/vrs`;
- the `detection-publisher` sidecar can publish to Redis Streams;
- API and console services respond through Kubernetes Services.

This is a platform validation. Detector acceptance still requires parity,
dataset evaluation, throughput, queue-drop, verifier-latency, and evidence
retention checks on target hardware and target camera data.

from typing import Any

import httpx

RUNPOD_GQL = "https://api.runpod.io/graphql"


async def gql(api_key: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            RUNPOD_GQL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables or {}},
        )
        if not r.is_success:
            body = (r.text or "").strip()
            if len(body) > 600:
                body = f"{body[:600]}..."
            raise RuntimeError(f"RunPod GraphQL HTTP {r.status_code}: {body or r.reason_phrase}")
        body = r.json()
        if "errors" in body:
            msg = body["errors"][0].get("message", str(body["errors"]))
            raise RuntimeError(msg)
        return body.get("data") or {}


async def fetch_myself(api_key: str) -> dict[str, Any]:
    q = """
    query {
      myself {
        id
        email
        clientBalance
        authId
      }
    }
    """
    data = await gql(api_key, q)
    return data.get("myself") or {}


async def fetch_gpu_types(api_key: str) -> list[dict[str, Any]]:
    # stockStatus / maxUnreservedGpuCount come from lowestPrice with per-cloud input.
    # maxGpuCount* are aggregate pool hints on GpuType (no extra lowestPrice input).
    q_full = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        securePrice
        communityPrice
        maxGpuCountCommunityCloud
        maxGpuCountSecureCloud
        lowestPriceCommunity: lowestPrice(input: { gpuCount: 1, secureCloud: false }) {
          minimumBidPrice
          uninterruptablePrice
          stockStatus
          maxUnreservedGpuCount
        }
        lowestPriceSecure: lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
          minimumBidPrice
          uninterruptablePrice
          stockStatus
          maxUnreservedGpuCount
        }
      }
    }
    """
    q_mid = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        securePrice
        communityPrice
        maxGpuCountCommunityCloud
        maxGpuCountSecureCloud
      }
    }
    """
    q_min = """
    query {
      gpuTypes {
        id
        displayName
        memoryInGb
        secureCloud
        communityCloud
        securePrice
        communityPrice
      }
    }
    """
    data: dict[str, Any] = {}
    try:
        data = await gql(api_key, q_full)
    except RuntimeError:
        try:
            data = await gql(api_key, q_mid)
        except RuntimeError:
            data = await gql(api_key, q_min)
    gpus = data.get("gpuTypes") or []
    return [g for g in gpus if g.get("id") and str(g["id"]).lower() != "unknown"]


async def deploy_pod(
    api_key: str,
    *,
    name: str,
    gpu_type_id: str,
    image_name: str,
    cloud_type: str,
    env_list: list[tuple[str, str]],
    container_disk_gb: int = 100,
    volume_gb: int = 100,
    ports: str = "8888/http,11434/http,4000/http",
) -> dict[str, Any]:
    q = """
    mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        name
        desiredStatus
        imageName
        costPerHr
        machineId
        gpuCount
      }
    }
    """
    env = [{"key": k, "value": v} for k, v in env_list]
    variables = {
        "input": {
            "name": name,
            "gpuTypeId": gpu_type_id,
            "gpuCount": 1,
            "imageName": image_name,
            "cloudType": cloud_type,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace",
            "ports": ports,
            "supportPublicIp": True,
            "env": env,
        }
    }
    data = await gql(api_key, q, variables)
    pod = data.get("podFindAndDeployOnDemand")
    if not pod:
        raise RuntimeError("RunPod did not return a pod")
    return pod


async def fetch_pod(api_key: str, pod_id: str) -> dict[str, Any]:
    q_primary = """
    query Pod($podId: String!) {
      pod(input: { podId: $podId }) {
        id
        name
        desiredStatus
        costPerHr
        gpuCount
        vcpuCount
        memoryInGb
        containerDiskInGb
        volumeInGb
        cpuFlavorId
        runtime {
          uptimeInSeconds
          gpus {
            id
            gpuUtilPercent
            memoryUtilPercent
          }
        }
        latestTelemetry {
          cpuUtilization
          memoryUtilization
          averageGpuMetrics {
            id
            percentUtilization
            memoryUtilization
          }
          individualGpuMetrics {
            id
            percentUtilization
            memoryUtilization
          }
        }
      }
    }
    """
    q_fallback = """
    query Pod($podId: String!) {
      pod(input: { podId: $podId }) {
        id
        name
        desiredStatus
        costPerHr
        gpuCount
        vcpuCount
        memoryInGb
        runtime {
          uptimeInSeconds
          gpus {
            id
            gpuUtilPercent
            memoryUtilPercent
          }
        }
        latestTelemetry {
          averageGpuMetrics {
            percentUtilization
            memoryUtilization
          }
          individualGpuMetrics {
            percentUtilization
            memoryUtilization
          }
        }
      }
    }
    """
    q_filter_var = """
    query Pod($input: PodFilter) {
      pod(input: $input) {
        id
        name
        desiredStatus
        costPerHr
        gpuCount
        vcpuCount
        memoryInGb
      }
    }
    """
    q_myself = """
    query MyselfPods {
      myself {
        pods {
          id
          name
          desiredStatus
          costPerHr
          gpuCount
          vcpuCount
          memoryInGb
        }
      }
    }
    """
    # RunPod schema can differ across API versions/accounts; fall back gracefully.
    last_error: Exception | None = None
    try:
        data = await gql(api_key, q_primary, {"podId": pod_id})
    except Exception as e:
        last_error = e
        try:
            data = await gql(api_key, q_fallback, {"podId": pod_id})
        except Exception as e2:
            last_error = e2
            try:
                data = await gql(api_key, q_filter_var, {"input": {"podId": pod_id}})
            except Exception as e3:
                last_error = e3
                data = await gql(api_key, q_myself)

    pod = data.get("pod") or {}
    if not pod:
        pods = ((data.get("myself") or {}).get("pods") or []) if isinstance(data, dict) else []
        pod = next((p for p in pods if str(p.get("id")) == str(pod_id)), {})
    if not pod and last_error:
        raise last_error

    runtime = pod.get("runtime") or {}
    telemetry = pod.get("latestTelemetry") or {}

    # Normalize commonly consumed fields for callers.
    if pod.get("uptimeSeconds") is None:
        if runtime.get("uptimeSeconds") is not None:
            pod["uptimeSeconds"] = runtime.get("uptimeSeconds")
        elif runtime.get("uptimeInSeconds") is not None:
            pod["uptimeSeconds"] = runtime.get("uptimeInSeconds")

    def _gpu_telemetry_row() -> dict[str, Any] | None:
        ind = telemetry.get("individualGpuMetrics") or []
        if ind and isinstance(ind[0], dict):
            return ind[0]
        avg = telemetry.get("averageGpuMetrics")
        if isinstance(avg, dict) and avg:
            return avg
        return None

    # Pod.hardware `gpus` are ids only; utilization lives on Pod.runtime.gpus (PodRuntimeGpus).
    rt_gpus = [g for g in (runtime.get("gpus") or []) if isinstance(g, dict)]
    if rt_gpus:
        pod["gpus"] = [
            {
                "id": g.get("id"),
                "gpuUtilPercent": g.get("gpuUtilPercent"),
                "memoryUtilPercent": g.get("memoryUtilPercent"),
            }
            for g in rt_gpus
        ]
    else:
        gm = _gpu_telemetry_row()
        if gm:
            pod["gpus"] = [
                {
                    "id": gm.get("id") or "telemetry",
                    "gpuUtilPercent": gm.get("percentUtilization"),
                    "memoryUtilPercent": gm.get("memoryUtilization"),
                }
            ]
        else:
            gpus = pod.get("gpus") or []
            gpu0 = gpus[0] if gpus else {}
            if not gpus and telemetry.get("gpuUtilPercent") is not None:
                pod["gpus"] = [
                    {
                        "id": "telemetry",
                        "gpuUtilPercent": telemetry.get("gpuUtilPercent"),
                        "memoryUtilPercent": telemetry.get("memoryUtilPercent"),
                    }
                ]
            elif gpus and gpu0:
                gm2 = _gpu_telemetry_row()
                if gm2:
                    if gpu0.get("gpuUtilPercent") is None and gm2.get("percentUtilization") is not None:
                        gpu0["gpuUtilPercent"] = gm2.get("percentUtilization")
                    if (
                        gpu0.get("memoryUtilPercent") is None
                        and gm2.get("memoryUtilization") is not None
                    ):
                        gpu0["memoryUtilPercent"] = gm2.get("memoryUtilization")
    return pod

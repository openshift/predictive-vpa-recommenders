import recommender_config
from recommender.Pando import *
from recommender.PromCrawler import *


def selectsRecommender(vpas, recommender_name):
    selected_vpas = []
    for vpa in vpas["items"]:
        vpa_spec = vpa["spec"]
        if "recommenders" not in vpa_spec.keys():
            continue
        else:
            print(vpa_spec)
            for recommender in vpa_spec["recommenders"]:
                if recommender["name"] == recommender_name:
                    selected_vpas.append(vpa)

    return selected_vpas


# resource2str converts a resource (CPU, Memory) value to a string
def resource2str(resource, value):
    if resource.lower() == "cpu":
        if value < 1:
            return str(int(value * 1000)) + "m"
        else:
            return str(value)
    # Memory is in bytes
    else:
        if value < 1024:
            return str(value) + "B"
        elif value < 1024 * 1024:
            return str(int(value / 1024)) + "k"
        elif value < 1024 * 1024 * 1024:
            return str(int(value / 1024 / 1024)) + "Mi"
        else:
            return str(int(value / 1024 / 1024 / 1024)) + "Gi"


# Convert a resource (CPU, Memory) string to a float value
def str2resource(resource, value):
    if type(value) is str:
        if resource.lower() == "cpu":
            if value[-1] == "m":
                return float(value[:-1]) / 1000
            else:
                return float(value)
        else:
            if value[-1].lower() == "b":
                return float(value[:-1])
            elif value[-1].lower() == "k":
                return float(value[:-1]) * 1024
            elif value[-2:].lower() == "mi":
                return float(value[:-2]) * 1024 * 1024
            elif value[-2:].lower() == "gi":
                return float(value[:-2]) * 1024 * 1024 * 1024
            else:
                return float(value)
    else:
        return value

def get_target_containers(corev1_client, target_namespace, target_ref):
    target_pods = corev1_client.list_namespaced_pod(namespace=target_namespace, label_selector="app=" + target_ref["name"])

    # Retrieve the target containers
    target_containers = []
    for pod in target_pods.items:
        for container in pod.spec.containers:
            if container.name not in target_containers:
                target_containers.append(container.name)

    return target_containers


def get_max_trace_among_pods(traces):
    max_traces = {}
    for container in traces.keys():
        max_traces[container] = {}
        for resource_type in traces[container].keys():
            max_traces[container][resource_type] = {}
            for pod in traces[container][resource_type].keys():
                cur_trace = traces[container][resource_type][pod]
                for data in cur_trace:
                    if data[0] not in max_traces[container][resource_type].keys():
                        max_traces[container][resource_type][data[0]] = float(data[1])
                    else:
                        max_traces[container][resource_type][data[0]] = max(float(data[1]), max_traces[container][resource_type][data[0]])

    return max_traces

def get_recommendation(vpa, corev1, prom_client):
    """
    This function takes a VPA and returns a list of recommendations
    """
    # Get the VPA spec
    vpa_spec = vpa["spec"]

    # example target_ref {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'name': 'hamster'}
    target_ref = vpa_spec["targetRef"]
    print(target_ref)

    # Retrieve the target pods
    if "namespace" in target_ref.keys():
        target_namespace = target_ref["namespace"]
    else:
        target_namespace = recommender_config.DEFAULT_NAMESPACE

    # Build the prometheus query for the target resources of target containers in target pods
    namespace_query = "namespace=\'" + target_namespace + "\'"

    # Get the target containers
    target_containers = get_target_containers(corev1, target_namespace, target_ref)

    # Get the target container traces
    traces = {}
    predictions = {}
    recommendations = []

    for containerPolicy in vpa_spec["resourcePolicy"]["containerPolicies"]:
        container_queries = []
        if containerPolicy["containerName"] != "*":
            container_query = "container='" + containerPolicy["containerName"] + "'"
            container_queries.append(container_query)
        else:
            for container in target_containers:
                container_query = "container='" + container + "'"
                container_queries.append(container_query)

        controlled_resources = containerPolicy["controlledResources"]
        max_allowed = containerPolicy["maxAllowed"]
        min_allowed = containerPolicy["minAllowed"]
        for resource in controlled_resources:
            if resource.lower() == "cpu":
                resource_query = "rate(container_cpu_usage_seconds_total{%s}[1m])"
            elif resource.lower() == "memory":
                resource_query = "container_memory_usage_bytes{%s}"
            else:
                print("Unsupported resource: " + resource)
                break

            # Retrieve the metrics for target containers in all pods
            for container_query in container_queries:
                # Retrieve the metrics for the target container
                query_index = namespace_query + "," + container_query

                query = resource_query % (query_index)
                print(query)

                # Retrieve the metrics for the target container
                traces = prom_client.get_promdata(query, traces, resource)

        # Merge the traces for the target container belonging to the same pods that restarted
        max_traces = get_max_trace_among_pods(traces)

        # Apply the forecasting & recommendation algorithms
        for container in max_traces.keys():
            for resource_type in max_traces[container].keys():
                cur_max_trace = max_traces[container][resource_type].items()
                metrics = np.array(list(cur_max_trace), dtype=float)
                metrics = np.sort(metrics, axis=0)
                predictions = construct_nested_dict(predictions, container, resource_type)
                forecast_window = int(recommender_config.FORECASTING_WINDOW / recommender_config.SAMPLING_PERIOD)
                forecast, prov, labels = pando_recommender(metrics[:, 1],
                                                           recommender_config.TREE,
                                                           window=forecast_window,
                                                           limit=recommender_config.LIMIT)
                predictions[container][resource_type] = forecast.tolist()

        for container in predictions.keys():
            container_recommendation = {"containerName": container, "lowerBound": {}, "target": {},
                                        "uncappedTarget": {}, "upperBound": {}}
            for resource in predictions[container].keys():
                all_pod_predictions = predictions[container][resource]

                lower_bound = np.percentile(all_pod_predictions, recommender_config.LOWERBOUND_PERCENTILE)
                uncapped_target = np.percentile(all_pod_predictions, recommender_config.TARGET_PERCENTILE)
                upper_bound = np.percentile(all_pod_predictions, recommender_config.UPPERBOUND_PERCENTILE)

                # If the target is below the lowerbound, set it to the lowerbound
                min_allowed_value = str2resource(resource, min_allowed[resource])
                max_allowed_value = str2resource(resource, max_allowed[resource])
                if uncapped_target < min_allowed_value:
                    target = min_allowed_value
                # If the target is above the upperbound, set it to the upperbound
                elif uncapped_target > max_allowed_value:
                    target = max_allowed_value
                # Otherwise, set it to the uncapped target
                else:
                    target = uncapped_target

                # Convert CPU/Memory values to millicores/bytes
                container_recommendation["lowerBound"][resource] = resource2str(resource, lower_bound)
                container_recommendation["target"][resource] = resource2str(resource, target)
                container_recommendation["uncappedTarget"][resource] = resource2str(resource, uncapped_target)
                container_recommendation["upperBound"][resource] = resource2str(resource, upper_bound)

            recommendations.append(container_recommendation)
    return recommendations

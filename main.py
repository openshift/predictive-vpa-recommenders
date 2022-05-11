from recommender.recommender import *
from recommender.PromCrawler import *
from kubernetes import client, config
from kubernetes.client.rest import ApiException

DOMAIN = "autoscaling.k8s.io"
VPA_NAME = "verticalpodautoscaler"
VPA_PLURAL = "verticalpodautoscalers"
VPA_CHECKPOINT_NAME = "verticalpodautoscalercheckpoint"
VPA_CHECKPOINT_PLURAL = "verticalpodautoscalercheckpoints"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        config.load_kube_config()

    # Get the api instance to interact with the cluster
    api_client = client.api_client.ApiClient()
    v1 = client.ApiextensionsV1Api(api_client)
    corev1 = client.CoreV1Api(api_client)
    crds = client.CustomObjectsApi(api_client)
    resource_version = ''

    # Initialize the prometheus client
    prom_client = PromCrawler(recommender_config.PROM_URL)

    # Get the VPA CRD
    current_crds = [x['spec']['names']['kind'].lower() for x in v1.list_custom_resource_definition().to_dict()['items']]
    if VPA_NAME not in current_crds:
        print("VerticalPodAutoscaler CRD is not created!")
        exit(-1)

    while True:
        vpas = crds.list_cluster_custom_object(group=DOMAIN, version="v1", plural=VPA_PLURAL)
        selectedVpas = selectsRecommender(vpas, recommender_config.RECOMMENDER_NAME)

        prom_client.update_period(recommender_config.FORECASTING_SIGHT)

        # Retrieve the container metrics for all deployments managed by all vpas
        for vpa in selectedVpas:
            vpa_name = vpa["metadata"]["name"]
            vpa_namespace = vpa["metadata"]["namespace"]
            recommendations = get_recommendation(vpa, corev1, prom_client)

            if not recommendations:
                print("No new recommendations obtained, so skip updating the vpa object {}".format(vpa_name))
                continue

            # Update the recommendations.
            patched_vpa = {"recommendation": {"containerRecommendations": recommendations}}
            body = {"status": patched_vpa}
            vpa_api = client.CustomObjectsApi()

            # Update the VPA object
            # API call doc: https://github.com/kubernetes-client/python/blob/master/kubernetes/docs/CustomObjectsApi.md#patch_namespaced_custom_object
            try:
                vpa_updated = vpa_api.patch_namespaced_custom_object(group=DOMAIN, version="v1", plural=VPA_PLURAL, namespace=vpa_namespace, name=vpa_name, body=body)
                print("Successfully patched VPA object with the recommendation: %s" % vpa_updated['status']['recommendation']['containerRecommendations'])
            except ApiException as e:
                print("Exception when calling CustomObjectsApi->patch_namespaced_custom_object: %s\n" % e)

        time.sleep(recommender_config.SLEEP_WINDOW)








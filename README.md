# predictive-vpa-recommenders
The repo includes a set of Vertical Pod Autoscaler (VPA) recommenders pluggable with the default VPA on OpenShift 4.11+.

## Prerequisites
- Kubernetes 1.22+ or OpenShift 4.11
- [Kubernetes Vertical Pod Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler) or [OpenShift Vertical Pod Autoscaler Operator](https://docs.openshift.com/container-platform/4.9/nodes/pods/nodes-pods-vertical-autoscaler.html)
- [Enabling monitoring for user-defined projects (For OpenShift only)](https://docs.openshift.com/container-platform/4.11/monitoring/enabling-monitoring-for-user-defined-projects.html#accessing-metrics-from-outside-cluster_enabling-monitoring-for-user-defined-projects)

## Installation
### Install on OpenShift 4.11+
1. Install the VPA Operator from OperatorHub
![alt text](https://github.com/openshift/predictive-vpa-recommenders/blob/main/docs/imgs/vpa-install.png?raw=true)

2. Create an example workload and its VPA cr that uses default recommender.
```bash
oc apply -f manifests/examples/test-periodic-recommender.yaml
```

3. Create a same workload managed by a VPA CR that uses the predictive recommender.
```bash
oc apply -f manifests/examples/pando-test-periodic-recommender.yaml
````

4. Create a SA and corresponding rolebindings for the recommender to access the metrics.
```bash
oc apply -f manifests/openshift/recommender-sa.yaml
```
5. Grant the recommender SA the permission to access the metrics.
```bash
oc adm policy add-cluster-role-to-user cluster-monitoring-view -z predictive-recommender -n openshift-vertical-pod-autoscaler
```

6. Obtain the thanos-querier service to allow prometheus data access for the predictive VPA recommender.
```bash
PROM_SECRET=`oc get secret -n openshift-vertical-pod-autoscaler | grep  predictive-recommender-token | head -n 1 | awk '{print $1 }'`
PROM_TOKEN=`echo $(oc get secret $PROM_SECRET -n openshift-vertical-pod-autoscaler -o json | jq -r '.data.token') | base64 -d`
PROM_HOST=`oc get route thanos-querier -n openshift-monitoring -o json | jq -r '.spec.host'`
```
Replace the PROM_TOKEN and PROM_HOST with their values in `manifests/openshift/pando-recommender.yaml`.
Then, replace the `${DEFAULT_NAMESPACE}` by `default` or any other namespace where the VPA is deployed.

7. Deploy the predictive VPA recommender
```bash
oc create -f manifests/openshift/pando-recommender.yaml
```


### Install on Kubernetes 1.22+

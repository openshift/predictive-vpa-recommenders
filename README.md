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
1. Clone the repository build and push to an accessible docker registry, the generated image is used [here](./manifests/kubernetes/pando-recommender-deployment.yaml) as the deployment's image.
```bash
git clone https://github.com/openshift/predictive-vpa-recommenders.git

docker build -t registry-name.com/some_path/pred-vpa-rec:latest .

docker push registry-name.com/some_path/pred-vpa-rec:latest
```
2. Update the prometheus settings (PROM_URL) in [here](./recommender_config.yaml) and [here](./manifests/kubernetes/pando-recommender-deployment.yaml) to match your local Kubernetes settings 

3. Install VPA in Kubernetes
To install VPA, download the source code of VPA (for example with git clone https://github.com/kubernetes/autoscaler.git) and run 
the following command inside the vertical-pod-autoscaler directory: 

```./hack/vpa-up.sh```

For more details visit https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler

2. Install pando recommender
```bash
cd predictive-vpa-recommenders
kubectl apply -f manifests/kubernetes/pando-recommender-deployment.yaml
```

3. Deploy an example workload and its VPA cr that uses default recommender.
```bash
kubectl apply -f manifests/examples/pando-test-periodic-deployment.yaml
```

4. Check if pando recommender is running, if so you should see some stats as below
```bash  
kubectl -n kube-system logs [pando_pod_name] --follow
```
```Successfully patched VPA object with the recommendation: [{'containerName': 'pando-test-periodic', 'lowerBound': {'cpu': '100m', 'memory': '50Mi'}, 'target': {'cpu': '100m', 'memory': '50Mi'}, 'uncappedTarget': {'cpu': '10m', 'memory': '8Mi'}, 'upperBound': {'cpu': '100m', 'memory': '50Mi'}}]```

5. Check default VPA recommender, the count of VPA objects should be 0
```bash
kubectl logs [vpa-recommender_pod_name] -n kube-system --follow
import time

import requests
import os
import pandas as pd
import numpy as np
import json

try:
    from .roundTime import *
except ImportError:
    from roundTime import *

requests.packages.urllib3.disable_warnings()


class PromCrawler:
    now = None
    start = None
    prom_address = "http://127.0.0.1:9090"
    crawling_period = 3600
    prom_token = None
    step = '15s'
    chunk_sz = 900

    def __init__(self, prom_address=None, prom_token=None):
        self.prom_address = prom_address or os.getenv("PROM_HOST")
        self.prom_token = prom_token or os.getenv("PROM_TOKEN")

        if not self.prom_address or not self.crawling_period:
            raise ValueError(
                "Please appropriately configure environment variables $PROM_HOST, $PROM_TOKEN, $CRAWLING_PERIOD to successfully run the crawler and profiler!")

    def update_period(self, crawling_period):
        self.crawling_period = crawling_period
        self.now = int(roundTime(dt=datetime.now()))
        self.start = int(self.now - self.crawling_period)  # last hour
        self.end = self.now

    def get_current_time(self):
        current_time_str = datetime.fromtimestamp(self.now).strftime("%I:%M:%S")
        return current_time_str

    def fetch_data_range(self, my_query, start, end):
        try:
            if self.prom_token:
                headers = {"content-type": "application/json; charset=UTF-8",
                           'Authorization': 'Bearer {}'.format(self.prom_token)}
            else:
                headers = {"content-type": "application/json; charset=UTF-8"}
            response = requests.get('{0}/api/v1/query_range'.format(self.prom_address),
                                    params={'query': my_query, 'start': start, 'end': end, 'step': self.step},
                                    headers=headers, verify=False)

        except requests.exceptions.RequestException as e:
            print(e)
            return None

        try:
            if response.json()['status'] != "success":
                print("Error processing the request: " + response.json()['status'])
                print("The Error is: " + response.json()['error'])
                return None

            results = response.json()['data']['result']

            if (results is None):
                # print("the results[] came back empty!")
                return None

            length = len(results)
            if length > 0:
                return results
            else:
                # print("the results[] has no entries!")
                return None
        except:
            print(response)
            return None

    def fetch_data_range_in_chunks(self, my_query):
        all_metric_history = []
        for cur_start in range(self.start, self.end, self.chunk_sz):
            cur_end = cur_start + self.chunk_sz

            trials = 0
            cur_metric_history = None
            while (cur_metric_history == None) and (trials < 3):
                cur_metric_history = self.fetch_data_range(my_query, cur_start, cur_end)
                trials += 1

            if cur_metric_history is None:
                continue

            all_metric_history += cur_metric_history

        return all_metric_history

    def get_promdata(self, query, traces, resourcetype):
        cur_trace = self.fetch_data_range_in_chunks(query)

        # Convert the prometheus data to a list of floats
        try:
            metric_obj_attributes = cur_trace[0]["metric"].keys()
        except:
            print("There are no data points for metric query {}.".format(query))
            return traces

        try:
            metric_obj_attributes = cur_trace[0]["metric"].keys()
        except:
            print("There are no data points for metric query {}.".format(query))
            return traces

        pod_key_name = get_key_name("pod", metric_obj_attributes)
        container_key_name = get_key_name("container", metric_obj_attributes)
        ns_key_name = get_key_name("namespace", metric_obj_attributes)
        if ns_key_name == "":
            ns_key_name = get_key_name("ns", metric_obj_attributes)

        if pod_key_name == "" or container_key_name == "" or ns_key_name == "":
            print(
                "[Warning] The metric object returned from Prometheus query {} does not have required attribute tags.".format(query))
            print("[Warning] The following attributes to the metric should not be empty.")
            print("[Warning] - pod attribute name: {}".format(pod_key_name))
            print("[Warning] - container attribute name: {}".format(container_key_name))
            print("[Warning] - namespace attribute name: {}".format(ns_key_name))

        for metric_obj in cur_trace:
            try:
                pod = metric_obj["metric"][pod_key_name]
            except:
                # print("No namespace or pod names in the timeseries data.")
                continue

            try:
                container = metric_obj["metric"][container_key_name]

                if container == "POD":
                    # print("The current metric is for the pod.")
                    continue
            except:
                continue

            metrics = metric_obj['values']
            traces = construct_nested_dict(traces, container, resourcetype, pod)
            traces[container][resourcetype][pod] += metrics
        return traces


# Other uitlity functions for the PromCrawler class
def construct_nested_dict(traces_dict, container, resourcetype, pod=None):
    if pod is None:
        if container not in traces_dict.keys():
            traces_dict[container] = {resourcetype: []}
        elif resourcetype not in traces_dict[container].keys():
            traces_dict[container][resourcetype] = []
    else:
        if container not in traces_dict.keys():
            traces_dict[container] = {resourcetype: {pod: []}}
        elif resourcetype not in traces_dict[container].keys():
            traces_dict[container][resourcetype] = {pod: []}
        elif pod not in traces_dict[container][resourcetype].keys():
            traces_dict[container][resourcetype][pod] = []

    return traces_dict


def get_key_name(attribute, klist):
    keys = [kname for kname in klist if attribute in kname.lower()]
    if len(keys) > 0:
        return keys[0]
    else:
        return ""
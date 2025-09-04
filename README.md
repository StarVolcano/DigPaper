# DigPaper
DigPaper: RAG-based Paper Reading Agent

Coming soon...
## Getting Started

### Step1: Deploy Nougat and LayoutLMv3 API sevice

Please refer to the offical repository for [Nougat](https://github.com/facebookresearch/nougat) and [layoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3).

For Nougat, api service can be deployed with the following command:

```bash
$ nougat_api
```

The default api link is http://127.0.0.1:8503/predict/.

For LayoutLMv3, no offical api service is available, so we provide a simple [script](./layoutlm_api/app.py) to deploy the api service. 

First,  `/home/huxc/paper_agent/unilm/layoutlmv3/examples/object_detection`

First,  

### Step2: Deploy DigPaper
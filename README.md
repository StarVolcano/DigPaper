# DigPaper:RAG-based Paper Reading Agent
DigPaper: RAG-based Paper Reading Agent

## Getting Started

### Step1: Deploy Nougat and LayoutLMv3 API sevice

Please refer to the offical repository for [Nougat](https://github.com/facebookresearch/nougat) and [layoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3).

For Nougat, api service can be deployed with the following command:

```bash
$ nougat_api
```

The default api link is http://127.0.0.1:8503/predict/.

For LayoutLMv3, no offical api service is available, so we provide a simple [script](./layoutlm_api/app.py) to deploy the api service. 

First, place the three files (`inference.py`, `app.py`, and `api.sh`) from the folder `layoutlm_api/` into the `/home/huxc/paper_agent/unilm/layoutlmv3/examples/object_detection/` folder. 

Second, change the `weights` (line 20) in `app.py` to the path of downloaded pretrianed model.

Third, run the script `api.sh` to start the api service at http://127.0.0.1:8001/predict_batch/.

### Step2: Deploy DigPaper

Create conda environment for DigPaper api:
```bash
$ conda env create -f env/digpaper.yml
```

Modified the configation file `config.py` to set the api link of Nougat and LayoutLMv3. 
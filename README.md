# Video registration of concrete samples

## Requirements

- Python 3.11
- GPU with `cuda` support
- (optional) 50GB of disc space for the dataset
- git lfs installed

There is a jupyter notebook `demo.ipynb` that goes through the video processing pipeline. In order to be able to use it, you will need to create a Python venv and download the required packages. This can be done with the following

```bash
# Downloads the LightGlue submodule
git submodule update --init
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This should install everything that is required. The dataset can be downloaded with the following:
> [!WARNING]
> The whole dataset spans around 50GB of disc space. You can download a single video from the repository's page here: [here.](https://huggingface.co/datasets/research-centre-rez/concrete-samples/blob/main/3A.MP4) 

```bash
git clone https://huggingface.co/datasets/research-centre-rez/concrete-samples
cd concrete-samples
git lfs pull
```

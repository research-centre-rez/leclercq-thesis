# Video registration of concrete samples

This is a repository that contains the source code of Erik Leclercq's master's thesis. Due to the large size of the dataset, it is hosted on HuggingFace, link to the dataset [can be found here](https://huggingface.co/datasets/research-centre-rez/concrete-samples). While the dataset is public, you will still need to have a HuggingFace account to be able to access it.

The source code is placed in the `src/` directory. The dataset downloading script below will download the dataset to a `concrete-samples/` directory. The code was tested only on `Ubuntu 22.04 LTS`. 

## Requirements

- Python 3.11
- GPU with `cuda` support
- (optional) 50GB of disc space for the dataset
- git lfs installed, instruction on how to install git lfs [can be found here.](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)  

There is a jupyter notebook `demo.ipynb` that goes through the video processing pipeline. In order to be able to use it, you will need to create a Python venv and download the required packages. This can be done with the following

```bash
# Downloads the LightGlue submodule
git submodule update --init
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupytext src/demo.py --to ipynb
```

The dataset can be downloaded with the following bash script:
> [!WARNING]
> The whole dataset spans around 50GB of disc space. You can download a single video from the repository's page [here.](https://huggingface.co/datasets/research-centre-rez/concrete-samples/blob/main/3A.MP4). If you choose to do so, you will need to update the video path in `src/demo.ipynb`.

```bash
git clone https://huggingface.co/datasets/research-centre-rez/concrete-samples
cd concrete-samples
git lfs pull
```

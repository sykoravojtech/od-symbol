
<!-- PROJECT Title -->
<br />
<div align="center">
  <h1 align="center">Multi-Modal Deep Learning for Automated Schematic Analysis</h1>

  <p align="center">
    <b>Master's Thesis at The University of Tübingen</b>
  </p>
  <p align="center">
    <i>Author: Vojtěch Sýkora</i>
  </p>
</div>
<!-- ----- -->


<!-- ABOUT THE PROJECT -->
## About the Project
Object detection for automating the extraction of electric schematics remains affected by two main issues: severe class imbalance and a lack of good training data, forcing a substantial domain shift between messy handwritten and clean computer-generated schematics. This thesis tackles both. First, a new Raspberry Pi (RPi)
dataset of clean high-resolution CAD schematics is created with 22 images of 1675
annotated symbols, aimed at working with a domain shift from the CGHD hand-
drawn dataset of 3137 images and 246k annotations. Second, the Faster R-CNN
object detector from the Modular Graph Extraction pipeline is enhanced using
losses, architectural changes, hyperparameter tuning, and image transformations.
The combination of Focal and GIoU losses, followed by tuning, shows a 7.3% mAP
increase on CGHD and 3.7% on the RPi dataset, both compared to a strong baseline. Next, a pretrained VLM, Molmo-7B-D, falls short of 32.0% accuracy of the
Faster R-CNN with 17.5% accuracy on the same object detection task; however, it
shows basic understanding of the task. This work confirms that a carefully modified
specialist model beats a pretrained VLM for symbol detection, and releases a new
RPi dataset to help with cross-domain adaptation.

<!-- GETTING STARTED -->
# Getting started
## Cloning the repo
```
git clone https://github.com/sykoravojtech/od-symbol.git
cd od-symbol
```

## Installing libraries
All requirements in `pyproject.toml` 
black and isort not needed, they are for automatic code style formatting.
```
poetry install
```
I use python 3.11.12

## Download the CGHD dataset
### Download from zenodo
```
cd data
mkdir cghd_raw
cd cghd_raw
wget -O cghd-zenodo-14.zip "https://zenodo.org/records/14042961/files/cghd-zenodo-14.zip?download=1"
```

```
unzip cghd-zenodo-14.zip
cd ../..
```
and optionally remove the zip
```
rm data/cghd_raw/cghd-zenodo-14.zip
```


### Download RPi datasheets
download the pdf datasheets
```
python -m data.utils.rpi.scrape_rpi_datasheets
```
extract information and images from them and save it in a separate directory
```
python -m data.utils.rpi.extract_rpi_images
```
move prepared annotations to the rpi_pico_sample/ 
```
mv data/annotations/ data/rpi_pico_sample/drafter_31/
```
and remove the not needed datasheets
```
rm -rf data/rpi_datasheets
rm -rf data/rpi_images_filt
```

<!-- Running basics -->
## Inference
Done using `extraction.src.core.inference_OD`. If verbosity v=2, then output found in `extraction/output`.

single image
```
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_cghd.json -i data/cghd_sample/drafter_1/images/C1_D2_P1.jpg -v 0

poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_rpi.json -i data/rpi_pico_sample/drafter_31/images/48c9152c-0-1.jpg -v 0
```
whole dataset
```
poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_cghd.json --dir data/cghd_raw -v 0

poetry run python -m extraction.src.core.inference_OD -c extraction/config/object_detection_inf_rpi.json --dir data/rpi_pico_sample/ -v 0
```

## Training
Done using `extraction.src.core.finetuning`. Results found in `extraction/model/`.
```
poetry run python -m extraction.src.core.finetuning extraction/config/od_enh/od_focal-giou.json
```



<!-- Other stuff -->
# Miscellaneous info
## Render symbols as pngs
```
python -m converter.core.symbol
```

## Other ways to download CGHD
### Download CGHD from huggingface
```
sudo apt install git-lfs
```

```
git lfs install
git clone https://huggingface.co/datasets/lowercaseonly/cghd
```

### Load CGHD from huggingface
```
pip install datasets
from datasets import load_dataset
dataset = load_dataset("lowercaseonly/cghd")
print(dataset)
print(dataset["train"][0])  # View first sample
```

## Download poetry
```
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```
sudo apt install python3.11-tk
poetry env use python3.11tk

## Download Python 3.11
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11
```

## Cluster
```
qsub -I -l select=1:ncpus=4:ngpus=1:mem=24gb:scratch_ssd=2gb -l walltime=6:00:00
cd /storage/brno2/home/nademvit/vthesis/cg25
```

## LaTeX on Ubuntu 24 LTS
`sudo apt install texlive-full -y`

install the latex workshop extension in vscode

## Rendering symbols

python -m converter.core.symbol

## Extracting images from RPi datasheets
using `docling` python library

## Annotating/labeling images for object detection, instance segmentation ...
python library `labelme`

## GenAI
- docstrings were generated using Claude Sonnet 4 in Copilot
- comments for better readability were generated using Claude Sonnet 4 in Copilot

<!-- ORIGINAL REPO SECTION -->
# Circuitgraph (Original repository)
Circuitgraph is a software for extracting electrical graphs from handwritten and printed circuit diagram (schematic) images as well as understanding, explaining and refining them. This repository mainly serves as an aggregation point to get started conveniently. For more details, please refer to the README files of the individual submodules.

## Setup
First of all, clone this repo using:

```
git clone https://codeberg.org/Circuitgraph/Main.git circuitgraph
```

Go to the repo folder:

```
cd circuitgraph
```

and check out all submodules:

```
git submodule update --init
```

Install the dependencies:

```
pip install -r ui/requirements.txt
```

Dowload the CGHD dataset from [Zenodo](https://zenodo.org/record/8266951) or [Kaggle](https://www.kaggle.com/datasets/johannesbayer/cghd1152) and place the content of the zip file in the `gtdb-hd` folder.

## Desktop Application Usage
While beeing in the root folder of the `main` repository, run:

```
python3 -m ui.main
```
# Circuitgraph Extraction
This repository contains scripts for training and inferencing models that serve the process of extracting electrical graphs from raster images.

## Folder Structure
The folder structure of this repository is made up as follows:

```
extraction
│   README.md                  # This File
└───src                        # Code Base
│   └───core                   # Core Functionality
│   │   │   inference.py       # General Inferencing
│   │   │   package_laoder.py  # Class Loader
│   │   └   training.py        # Task-Agnostic Training Cycle
│   └───loader                 # Dataset Loader Classes
│   │   └   ...
│   └───models                 # Model Definition Classes
│   │   └   ...
│   └───processor              # Pre/Post Processors
│   │   └   ...
│   └───utils                  # Ausiliary Functions
│       └   ...
└───config                     # Setup Files for Training and Inferencing
│   └   ...
└───model                      # Trained Models
│   └   ...
└───loaders                    # Data Cache
    └   ...
```

## System Overview
The central elements of the architecture are `Dataloader`, `Processor` and `Model`, which are joinedly set up within a `Config File`. For training and inference, they are arranged differently.

### Training
During the training process, the `Dataloader` reads files from a ground-truth database and preprocesses them by a `Processor`, effectively serving as data source to provide `Sample` lists for `Model` training. In order to speed up the (potentially time-consuming) sample preparation procedure, the preprocessed samples are stored in a cache and the preprocessing is skipped if a cache already exists:

```
                              ┌─────────┐
                              │ Dataset │
                              └─────────┘
                                   ▲
                                   │ Load
                                   │
  ┌───────┐      Load       ┌────────────┐     Raw Data     ┌───────────┐
  │ Cache │ ◄────────────── │ Dataloader │ ───────────────► │ Processor │
  └───────┘ ◄────────────── └────────────┘ ◄─────────────── └───────────┘
                Create             │         Input, Target
                                   │
                                   │
                                   │ Samples
                                   │
                                   ▼
                               ┌───────┐
                               │ Model │
                               └───────┘
```

### Inference
During inference, dataloaders are not required and the processor utilizes the model directly:

```
                           ┌────────────────┐
  ┌──────────┐             │   Processor    │     Input      ┌───────┐
  │ Raw Data │ ──────────► │ pre_process()  │ ─────────────► │ Model │
  └──────────┘ ◄────────── │ post_process() │ ◄───────────── └───────┘
                           └────────────────┘    Prediction
```


### Config Files
Config files are places in `extraction/config`. Every config file and describes a setup that can be used for both model training and inference.

### Dataloaders
The task of a loader is to read data from a folder structure of a dataset and provide them to an aggregated processor for generating a list of `(input, target)` pairs (both `input` and `target` have to be of type `torch.Tensor`). Dataloader are intended to perform low-level operations only and are typically made up quite simple. Loaders are places under `extraction/src/loaders`. Dataloaders should always inherit from the `Dataloader` base class, which automatically caches preprocessed datasets.

### Models
Models, are defined as PyTorch Module in `extraction/src/models` and trained models are stored under `extraction/model`.

### Processors
The main task of a processor (placed in `extraction/src/processor`) is to provide functionality for handling **a single sample** ("one circuit", e.g. a pair of graph and image). More Precisely, the Processor is responsible for:

 - Data Augmentation
 - Generation of Model Input
   - Generation of a list of pairs of `torch.Tensor` that serve as model input and target during training
 - Interpretation of model's output

A Processor consumes one or more common data structures (e.g. Numpy Arrays holding an image or NetworkX graph structures) and returns:

 - Training: A list of pytorch tensor pairs (one tensor is used as model input, the other as model target)
 - Inference: A list of pyTorch tensors (which are all fed to the model)

### Utils
Simple encoder/decoder, accuracy metrics and other auxiliary functions are places in `extraction/src/utils`.

## Training
The Training Pipeline is designed in an universal and modular fashion, in which `model` and `dataloader` classes need to be combined by a `config` file. For example, in order to perform handwriting recognition training, the training cycle need to be called as follows:

```
python3 -m extraction.src.core.training extraction/config/text.json
```

## Inference
Similar to the training process, inference relies on the provision of a `config` file. Apart from that, an input image as well as an input graph have to be provided:

```
python3 -m extraction.src.core.inference extraction/config/text.json temp_x/image.png temp_x/graph.xml
```

## Extraction
To perform an end-to-end evaluation of the pipleline, perform:

```
python3 -m extraction.src.core.evaluation
```


## Developer Notes

 - In every `Processor` subclass
   - have the `pre_process()` and `post_process()` at the top of the class definition
   - `pre_process()` and `post_process()` should be concise
     - give structure
     - call other methods
     - occupying one page at max

## Inference of object detection
```
prp -m extraction.src.core.inference_single extraction/config/object_detection.json data/cghd_sample/drafter_1/images/C1_D2_P1.jpg data/cghd_sample/drafter_1/annotations/C1_D2_P1.xml
```

## copy models to metacentrum
```
scp -r /home/vojta/Documents/cg/cg25/extraction/model/baseline nademvit@onyx.metacentrum.cz:/storage/brno2/home/nademvit/vthesis/cg25/extraction/model/
```
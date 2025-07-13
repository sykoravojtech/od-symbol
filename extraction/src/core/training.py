"""training.py: Main Training Cycle"""

# System Imports
import json
import sys
from datetime import datetime
from os import mkdir
from os.path import isdir, join
from random import shuffle
from subprocess import run
from typing import Callable, Optional, Tuple

# Third-Party Imports
import torch
from torchinfo import summary
from tqdm import tqdm

# Project Imports
from extraction.src.core.package_loader import class_from_package, clsstr
from extraction.src.core.visualisation import debug_dump, plot_curve
from extraction.src.models.fasterrcnn import FasterRCNN

__author__ = "Johannes Bayer, Vojtěch Sýkora"
__copyright__ = "Copyright 2023-2024, DFKI"
__license__ = "CC"
__version__ = "0.0.1"
__email__ = "johannes.bayer@dfki.de, sykoravojtech01@gmail.com"
__status__ = "Prototype"


def save_config(config: Dict, save_path: str):
    with open(save_path, "w") as file:
        json.dump(config, file, indent=2)


def apply_set(
    sample_set,
    model,
    optimizer,
    scheduler,
    loss_fn: Callable,
    acc_fn: Callable,
    batch_size: int,
    device,
    is_training: bool,
    exp_path: str,
    vis_fn: Callable,
    debug: bool,
) -> Tuple[float, float]:
    """Performs a Batch-Wise Dataset Application to the Model for Training and Evaluation"""
    if isinstance(model, FasterRCNN):  # object_detection
        return apply_set_fasterrcnn(
            sample_set,
            model,
            optimizer,
            scheduler,
            loss_fn,
            acc_fn,
            batch_size,
            device,
            is_training,
            exp_path,
            vis_fn,
            debug,
        )
    else:  # text, text_lstm, rotation, rotation_ta
        return apply_set_rest(
            sample_set,
            model,
            optimizer,
            scheduler,
            loss_fn,
            acc_fn,
            batch_size,
            device,
            is_training,
            exp_path,
            vis_fn,
            debug,
        )


def apply_set_fasterrcnn(
    sample_set,
    model,
    optimizer,
    scheduler,
    loss_fn: Callable,
    acc_fn: Callable,
    batch_size: int,
    device,
    is_training: bool,
    exp_path: str,
    vis_fn: Callable,
    debug: bool,
) -> Tuple[float, float]:
    """Performs a Batch-Wise Dataset Application to the Model for Training and Evaluation"""

    loss_average = 0
    acc_average = 0
    loss = 0
    sample_indices = list(range(len(sample_set)))
    # print(f"\tSample indices: {sample_indices}")

    if is_training:
        shuffle(sample_indices)
        model.train()

    else:
        model.eval()

    if vis_fn is not None:
        print("\tvis_fn is not None")
        inputs = torch.stack([sample[0] for sample in sample_set])
        targets = torch.stack([sample[1] for sample in sample_set])
        preds = torch.zeros((len(sample_set),) + sample_set[0][1].shape)
        infos = [sample[2] for sample in sample_set]

    # print(
    #     f"\tlen(sample_set): {len(sample_set)}/{batch_size}= {len(sample_set) // batch_size}"
    # )
    for batch_nbr in tqdm(
        range(len(sample_set) // batch_size),
        desc=f"{'Train Batches     ' if is_training else 'Validation Batches'}",
    ):
        # print(f"\n\t==> batch_nbr: {batch_nbr} <==")
        batch_sample_indices = sample_indices[
            batch_nbr * batch_size : (batch_nbr + 1) * batch_size
        ]
        # print(f"\tbatch_sample_indices: {batch_sample_indices}")
        """
        OBJECT DETECTION
        batch = List[Tuple[Tensor of image, 
                           Dict of {boxes: list, labels: list}
                           Dict of {info: None}]]
        """
        batch = [sample_set[i] for i in batch_sample_indices]

        patch_img = []  # batch of images
        """
        target = [
            {
                "boxes": torch.tensor([[x1, y1, x2, y2], ...]),  # Ground truth boxes
                "labels": torch.tensor([class1, class2, ...])   # Ground truth labels
            }
        ]
        """
        # target = torch.zeros([batch_size] + list(batch[0][1].size()))
        target = []

        # print(f"\tbatch({len(batch)}|({batch[0][0].shape},{batch[0][1].shape})):")
        # print(f"First element type: {type(batch[0])}")
        # print(
        #     f"First element length (if tuple or list): {len(batch[0]) if isinstance(batch[0], (tuple, list)) else 'N/A'}"
        # )
        # print(f"{batch[0][0]}\n\t{batch[0][1]}")
        # print(f"sample info {batch[0][2]}")
        for sample_nbr, (sample_img, sample_target, sample_info) in enumerate(batch):
            # this was changed because images are different sizes so we cannot use a tensor
            patch_img.append(sample_img)
            target.append(sample_target)
            # TODO rename batch_XXX and add _info

        # print(f"{patch_img=}")
        # print(f"{target=}")
        patch_img = [img.to(device) for img in patch_img]  # Move images to GPU
        target = [
            {k: v.to(device) for k, v in t.items()} for t in target
        ]  # Move target dicts to GPU

        optimizer.zero_grad()

        # **Training Mode: Compute Loss**
        if is_training:
            loss_dict = model(
                patch_img, target
            )  # Model returns a dict of loss components
            loss = sum(loss_dict.values())  # Sum up all losses

            loss.backward()
            optimizer.step()
            pred = None  # No predictions during training

        # **Evaluation Mode: Get Predictions**
        else:
            """
            pred is a list (instead of a tensor) of dictionaries because not every image has the same number of bounding boxes
            len(pred) = batch_size
            pred[0]: dict(boxes  = torch.Tensor(n_bboxes,4) where 4 is (x_min, y_min, x_max, y_max)
                            labels = torch.Tensor(n_bboxes,)
                            scores = torch.Tensor(n_bboxes,) aka the confidence score
            """
            with torch.no_grad():
                pred = model(patch_img)  # Model returns predictions as a list of dicts

            # Print sample prediction
            # print("\tPredictions for first image in batch:")
            # print(
            #     f"\tEVAL1:{pred[0]=}"
            # )  # Example: {'boxes': tensor(...), 'labels': tensor(...), 'scores': tensor(...)}
            loss = torch.tensor(
                0.0, dtype=torch.float32, device=device
            )  # No loss during evaluation

        if debug:
            debug_dump(patch_img, "input", exp_path)
            debug_dump(target, "target", exp_path)
            debug_dump(pred, "pred", exp_path)

        if vis_fn is not None:
            for pred_index, sample_index in enumerate(batch_sample_indices):
                preds[sample_index] = pred[pred_index].detach().clone()

        loss_average += loss.item() / batch_size

        if not is_training:
            acc_average += (
                sum([acc_fn(pred[i], target[i]) for i in range(batch_size)])
                / batch_size
            )

            # Compute mAP
            # mAP = compute_map(pred, target)  # Call compute_map
            # print(f"\tmAP: {mAP:.4f}")
    # end of for loop over batches

    loss_average = loss_average / (len(sample_set) // batch_size)
    acc_average = acc_average / (len(sample_set) // batch_size)

    if is_training:
        scheduler.step()

    if vis_fn is not None:
        if not isdir(join(exp_path, "eval")):
            mkdir(join(exp_path, "eval"))

        vis_fn(
            inputs,
            targets,
            preds,
            infos,
            join(exp_path, "eval"),
            "train" if is_training else "val",
        )

    print(
        f"{'Train     ' if is_training else 'Validation'}:    (only train)loss -> {loss_average:.10f}    (only val)acc -> {acc_average:.10f}"
    )

    return loss_average, acc_average


def apply_set_rest(
    sample_set,
    model,
    optimizer,
    scheduler,
    loss_fn: Callable,
    acc_fn: Callable,
    batch_size: int,
    device,
    is_training: bool,
    exp_path: str,
    vis_fn: Callable,
    debug: bool,
) -> Tuple[float, float]:
    """Performs a Batch-Wise Dataset Application to the Model for Training and Evaluation"""

    loss_average = 0
    acc_average = 0
    sample_indices = list(range(len(sample_set)))

    if is_training:
        shuffle(sample_indices)
        model.train()

    else:
        model.eval()

    if vis_fn is not None:
        inputs = torch.stack([sample[0] for sample in sample_set])
        targets = torch.stack([sample[1] for sample in sample_set])
        preds = torch.zeros((len(sample_set),) + sample_set[0][1].shape)
        infos = [sample[2] for sample in sample_set]

    for batch_nbr in tqdm(
        range(len(sample_set) // batch_size),
        desc=f"{'Train Batches     ' if is_training else 'Validation Batches'}",
    ):
        batch_sample_indices = sample_indices[
            batch_nbr * batch_size : (batch_nbr + 1) * batch_size
        ]
        batch = [sample_set[i] for i in batch_sample_indices]
        patch_img = torch.zeros(
            [batch_size] + list(batch[0][0].size()), dtype=torch.float32
        )
        target = torch.zeros([batch_size] + list(batch[0][1].size()))

        for sample_nbr, (sample_img, sample_target, sample_info) in enumerate(batch):
            patch_img[sample_nbr, :] = sample_img
            target[sample_nbr, :] = sample_target
            # TODO rename batch_XXX and add _info

        patch_img = patch_img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(patch_img)
        loss = loss_fn(pred, target)

        if is_training:
            loss.backward()
            optimizer.step()

        if debug:
            debug_dump(patch_img, "input", exp_path)
            debug_dump(target, "target", exp_path)
            debug_dump(pred, "pred", exp_path)

        if vis_fn is not None:
            for pred_index, sample_index in enumerate(batch_sample_indices):
                preds[sample_index] = pred[pred_index].detach().clone()

        loss_average += loss.item() / batch_size
        acc_average += (
            sum([acc_fn(pred[i], target[i]) for i in range(batch_size)]) / batch_size
        )

    loss_average = loss_average / (len(sample_set))
    acc_average = acc_average / (len(sample_set))

    if is_training:
        scheduler.step()

    if vis_fn is not None:
        if not isdir(join(exp_path, "eval")):
            mkdir(join(exp_path, "eval"))

        vis_fn(
            inputs,
            targets,
            preds,
            infos,
            join(exp_path, "eval"),
            "train" if is_training else "val",
        )

    print(
        f"{'Train     ' if is_training else 'Validation'}:    loss -> {loss_average:<22}    acc -> {acc_average:<18}"
    )

    return loss_average, acc_average


def train(
    model_cls,
    model_args,
    optim_cls,
    optim_args,
    scheduler_cls,
    scheduler_args,
    data_loader_cls,
    processor_cls,
    processor_args,
    loss_fn,
    acc_fn,
    set_train,
    set_val,
    db_path,
    save_path,
    batch_size,
    epoch_count,
    name,
    vis_fn,
    debug=False,
    model_path=None,
):
    """Performs the Model Training"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_folder = name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    mkdir(join(save_path, exp_folder))
    exp_folder = join(save_path, exp_folder)

    with open(join(exp_folder, "config.json"), "w") as config_file:
        config_file.write(
            json.dumps(
                {
                    "model_class": clsstr(model_cls),
                    "model_parameter": model_args,
                    "dataloader_class": clsstr(data_loader_cls),
                    "processor_class": clsstr(processor_cls),
                    "processor_parameter": processor_args,
                    "optimizer_class": clsstr(optim_cls),
                    "optimizer_parameter": optim_args,
                    "scheduler_class": clsstr(scheduler_cls),
                    "scheduler_parameter": scheduler_args,
                    "training_parameter": {
                        "drafters_set_train": set_train,
                        "drafters_set_val": set_val,
                        "drafter_set_test": [],
                        "db_path": db_path,
                        "batch_size": batch_size,
                        "epochs": epoch_count,
                        "name": name,
                        "loss_fn": str(loss_fn),
                        "acc_fn": str(acc_fn),
                        "vis_fn": str(vis_fn),
                        "debug": debug,
                    },
                },
                indent=2,
            )
        )

    print("Loading Train set")
    set_train = data_loader_cls(
        drafters=set_train,
        db_path=db_path,
        augment=True,
        processor_cls=processor_cls,
        processor_args=processor_args,
        debug=debug,
        caching=False,
    )

    print("Loading Validation set")
    set_val = data_loader_cls(
        drafters=set_val,
        db_path=db_path,
        augment=False,
        processor_cls=processor_cls,
        processor_args=processor_args,
        debug=debug,
        caching=False,
    )

    model = model_cls(**model_args)
    model.to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    optimizer = optim_cls(model.parameters(), **optim_args)
    scheduler = scheduler_cls(optimizer, **scheduler_args)

    if debug:
        print(summary(model))

    curve_train_loss = []
    curve_val_loss = []
    curve_train_acc = []
    curve_val_acc = []

    for epoch_nbr in range(epoch_count):
        print(f"\n==> Epoch {epoch_nbr} / {epoch_count} <==")

        train_loss, train_acc = apply_set(
            set_train,
            model,
            optimizer,
            scheduler,
            loss_fn,
            acc_fn,
            batch_size,
            device,
            is_training=True,
            exp_path=exp_folder,
            vis_fn=vis_fn,
            debug=debug,
        )

        val_loss, val_acc = apply_set(
            set_val,
            model,
            optimizer,
            scheduler,
            loss_fn,
            acc_fn,
            batch_size,
            device,
            is_training=False,
            exp_path=exp_folder,
            vis_fn=vis_fn,
            debug=debug,
        )

        curve_train_loss.append(train_loss)
        curve_val_loss.append(val_loss)
        curve_train_acc.append(train_acc)
        curve_val_acc.append(val_acc)

        plot_curve(
            {
                "Train Loss": curve_train_loss,
                "Val Loss": curve_val_loss,
                "Train Acc": curve_train_acc,
                "Val Acc": curve_val_acc,
            },
            epoch_count,
            exp_folder,
        )

        if val_acc == max(curve_val_acc):
            print("Saving Model...", end="", flush=True)
            torch.save(model.state_dict(), join(exp_folder, "model_state.pt"))
            print("Done.")

            if vis_fn is not None:
                if not isdir(join(exp_folder, "eval-best")):
                    mkdir(join(exp_folder, "eval-best"))

                run(
                    f"cp {join(exp_folder, 'eval', '*')} {join(exp_folder, 'eval-best')}",
                    shell=True,
                )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: One config JSON file needs to be provided")

    else:
        with open(sys.argv[1]) as json_file:
            config = json.loads(json_file.read())

        train(
            model_cls=class_from_package(config["model_class"]),
            model_args=config["model_parameter"],
            optim_cls=class_from_package(config["optimizer_class"]),
            optim_args=config["optimizer_parameter"],
            scheduler_cls=class_from_package(config["scheduler_class"]),
            scheduler_args=config["scheduler_parameter"],
            data_loader_cls=class_from_package(config["dataloader_class"]),
            processor_cls=class_from_package(config["processor_class"]),
            processor_args=config["processor_parameter"],
            loss_fn=class_from_package(config["training_parameter"]["loss_fn"])(),
            acc_fn=class_from_package(config["training_parameter"]["acc_fn"]),
            set_train=config["training_parameter"]["drafters_set_train"],
            set_val=config["training_parameter"]["drafters_set_val"],
            db_path=config["training_parameter"]["db_path"],
            save_path=join("extraction", "model"),
            batch_size=config["training_parameter"]["batch_size"],
            epoch_count=config["training_parameter"]["epochs"],
            name=config["training_parameter"]["name"],
            vis_fn=(
                class_from_package(config["training_parameter"]["vis_fn"])
                if "vis_fn" in config["training_parameter"]
                else None
            ),
            debug=config["training_parameter"]["debug"],
            model_path=config.get("model_path", None),
        )

"""
prp -m extraction.src.core.training extraction/config/object_detection.json

prp -m extraction.src.core.training extraction/config/full/object_detection.json

prp -m extraction.src.core.training extraction/config/object_detection_FINETUNE.json
"""

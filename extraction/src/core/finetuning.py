"""finetuning.py: Memory-Efficient Training Script with Model Loading Support"""

# System Imports
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime
from os import makedirs, mkdir, path
from os.path import isdir, join
from random import shuffle
from subprocess import run
from typing import Callable, Dict, List, Optional, Tuple

import psutil

# Third-Party Imports
import torch

# Project Imports
from extraction.src.core.package_loader import class_from_package, clsstr
from extraction.src.core.visualisation import debug_dump, plot_curve
from extraction.src.models.fasterrcnn import FasterRCNN
from extraction.src.models.fasterrcnn_addons.dilated_backbone import (
    attach_dilated_convs,
)
from extraction.src.models.fasterrcnn_addons.eca import attach_eca
from torchinfo import summary
from tqdm import tqdm

__author__ = "Vojtěch Sýkora"

K_SIZE = 5


def save_training_time(
    training_start_time, training_start_datetime, epoch_count, epoch_times, exp_path
):
    # Record training end time and calculate duration
    training_end_time = time.time()
    training_end_datetime = datetime.now()
    training_duration_seconds = training_end_time - training_start_time

    # Format duration for readability
    hours = int(training_duration_seconds // 3600)
    minutes = int((training_duration_seconds % 3600) // 60)
    seconds = training_duration_seconds % 60

    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    # Save training time information to file
    training_info = f"""Training Time Report
===================
Training started: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
Training ended:   {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}
Total duration:   {duration_str} (HH:MM:SS.sss)
Duration in seconds: {training_duration_seconds:.3f}
Total epochs: {epoch_count}
Average time per epoch: {training_duration_seconds/epoch_count:.3f} seconds

Epoch-by-Epoch Timing:
"""

    for i, epoch_time in enumerate(epoch_times):
        training_info += f"Epoch {i}: {epoch_time:.3f} seconds\n"

    if epoch_times:
        training_info += f"\nEpoch Statistics:\n"
        training_info += f"Fastest epoch: {min(epoch_times):.3f} seconds\n"
        training_info += f"Slowest epoch: {max(epoch_times):.3f} seconds\n"
        training_info += (
            f"Average epoch time: {sum(epoch_times)/len(epoch_times):.3f} seconds\n"
        )

    with open(join(exp_path, "training_time.txt"), "w") as f:
        f.write(training_info)

    print(f"Training completed in {duration_str}. Models saved to {exp_path}")
    print(f"Training time information saved to {join(exp_path, 'training_time.txt')}")


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
    """Memory-efficient implementation for Faster R-CNN dataset application"""
    # Force garbage collection before processing
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Use mixed precision for better memory efficiency on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    # Use a smaller batch size to reduce memory usage
    # batch_size = min(batch_size, 1)

    loss_average = 0
    acc_average = 0
    loss = 0
    sample_indices = list(range(len(sample_set)))

    if is_training:
        shuffle(sample_indices)
        model.train()
    else:
        model.eval()

    # Calculate number of batches
    num_batches = len(sample_set) // batch_size
    if num_batches == 0 and len(sample_set) > 0:
        num_batches = 1
        batch_size = len(sample_set)

    # Handle visualization data
    if vis_fn is not None:
        print("\tvis_fn is not None")
        inputs = []
        targets = []
        preds = []
        infos = []

        for sample in sample_set:
            inputs.append(sample[0])
            targets.append(sample[1])
            preds.append(torch.zeros_like(sample[1]))
            infos.append(sample[2])

    for batch_nbr in tqdm(
        range(num_batches),
        desc=f"{'Train Batches     ' if is_training else 'Validation Batches'}",
    ):
        # Clear cache between batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Get batch samples
        # Here we make sure even uneven sample set length with batch size works
        start_idx = batch_nbr * batch_size
        end_idx = min(start_idx + batch_size, len(sample_set))
        batch_sample_indices = sample_indices[start_idx:end_idx]
        batch = [sample_set[i] for i in batch_sample_indices]

        # Process batch
        patch_img = []
        target = []

        for sample_img, sample_target, sample_info in batch:
            patch_img.append(sample_img)
            target.append(sample_target)

        # Move data to device
        patch_img = [img.to(device) for img in patch_img]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        # Validate target data (TODO REMOVE)
        # skip_batch = False
        # for i, t in enumerate(target):
        #     if "boxes" in t and len(t["boxes"]) > 0:
        #         # Check for invalid box coordinates
        #         boxes = t["boxes"]
        #         if torch.any(torch.isnan(boxes)) or torch.any(torch.isinf(boxes)):
        #             print(
        #                 f"Warning: Invalid boxes detected in target {i}, skipping batch"
        #             )
        #             skip_batch = True
        #             break
        #         # Check if boxes are valid (x1 < x2, y1 < y2)
        #         if torch.any(boxes[:, 0] >= boxes[:, 2]) or torch.any(
        #             boxes[:, 1] >= boxes[:, 3]
        #         ):
        #             print(f"Warning: Invalid box format in target {i}, skipping batch")
        #             skip_batch = True
        #             break

        # # TODO REMOVE
        # if skip_batch:
        #     continue

        # Clear optimizer gradients
        optimizer.zero_grad()

        # Training step: compute loss
        if is_training:
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    loss_dict = model(patch_img, target)
                    loss = sum(loss_dict.values())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(patch_img, target)
                loss = sum(loss_dict.values())
                # print(f"{[t.item() for t in loss_dict.values()]}\nLoss={loss.item()=}")

                # Check for NaN/Inf in loss before backward pass
                if not torch.isfinite(loss):
                    print(f"\nWarning: Non-finite loss detected: {loss.item()}")
                    print(
                        f"Loss components: {[f'{k}: {v.item()}' for k, v in loss_dict.items()]}"
                    )
                    # Skip this batch to prevent NaN propagation
                    continue

                loss.backward()

                # Add gradient clipping to prevent gradient explosion
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # TODO REMOVE

                optimizer.step()

            pred = None  # No predictions during training
        else:  # Evaluating step: get predictions
            with torch.no_grad():
                pred = model(patch_img)

            loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Debug output
        if debug:
            debug_dump(patch_img, "input", exp_path)
            debug_dump(target, "target", exp_path)
            if pred is not None:
                debug_dump(pred, "pred", exp_path)

        # Store visualization data for later processing
        if vis_fn is not None and pred is not None:
            for pred_index, sample_index in enumerate(batch_sample_indices):
                preds[sample_index] = pred[pred_index]

        # Update statistics
        # Since the model returns the average loss per the whole batch we multiply it by batch len to get the total loss of the batch
        loss_average += loss.item() * len(batch)

        # Calculate accuracy for validation
        if not is_training and pred is not None:
            batch_acc = 0
            for i in range(len(pred)):
                batch_acc += acc_fn(pred[i], target[i])
            acc_average += batch_acc
            # print(f"{batch_acc=}")

        # Explicitly free the memory
        del patch_img, target
        if pred is not None:
            del pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Normalize metrics
    if num_batches > 0:
        loss_average = loss_average / len(sample_set)
        acc_average = acc_average / len(sample_set) if not is_training else 0

    if is_training:
        scheduler.step()

    # Process visualization
    if vis_fn is not None:
        if not isdir(join(exp_path, "eval")):
            mkdir(join(exp_path, "eval"))

        try:
            vis_fn(
                inputs,
                targets,
                preds,
                infos,
                join(exp_path, "eval"),
                "train" if is_training else "val",
            )
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")

    # print both loss and accuracy for both training and validation
    # print(
    #     f"{'Train     ' if is_training else 'Validation'}:    (only train)loss -> {loss_average:.10f}    (only val)acc -> {acc_average:.10f}"
    # )

    # When training, print loss_average and leave the acc_average slot empty,
    # and vice versa for validation.
    # Because the FasterRCNN model only gives us these values
    loss_str = f"loss -> {loss_average:.10f}" if is_training else "loss -> " + " " * 12
    acc_str = "acc -> " + " " * 15 if is_training else f"acc -> {acc_average:.10f}"
    print(
        f"{'Train     ' if is_training else 'Validation'}:    {loss_str}    {acc_str}"
    )

    return loss_average, acc_average


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
    """Router function to appropriate processing method based on model type"""
    # Check for FasterRCNN models (both original and enhanced)
    is_fasterrcnn = isinstance(model, FasterRCNN)

    if is_fasterrcnn:  # object_detection
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
    else:
        raise NotImplementedError("Only FasterRCNN models are supported in this script")


def get_cpu_info():
    """Get detailed CPU information for better debugging on CPU-only systems"""
    cpu_info = {
        "CPU": platform.processor(),
        "System": platform.system() + " " + platform.release(),
        "Total RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "Available RAM": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "CPU Count": os.cpu_count(),
        "CPU Usage": f"{psutil.cpu_percent()}%",
    }
    return cpu_info


def log_memory_stats(device):
    """Log memory statistics to help with debugging memory issues"""
    print("\n--- Memory Statistics ---")
    # System memory
    vm = psutil.virtual_memory()
    print(
        f"System RAM: {vm.used/1e9:.2f} GB used / {vm.total/1e9:.2f} GB total ({vm.percent}%)"
    )

    # GPU memory if available
    if device.type == "cuda":
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        print(
            f"GPU max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB"
        )
    print("------------------------\n")


def finetune(
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
    use_eca=False,
    use_dilated=False,
):
    """Memory-efficient training process with support for loading pretrained models"""
    # Initialize CUDA memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        # Add memory-saving options
        torch.backends.cudnn.deterministic = True

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print device information
    print(f"\n=== Device Information ===")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU architecture: {torch.cuda.get_device_capability(0)}")
        print(
            f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("Using CPU (No GPU available)")
        cpu_info = get_cpu_info()
        for key, value in cpu_info.items():
            print(f"{key}: {value}")
    print("=======================\n")

    # Log initial memory stats
    log_memory_stats(device)

    # Create experiment folder
    mid = "_finetune_" if model_path else "_train_"
    exp_folder = name + mid + datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-4]
    exp_path = join(save_path, exp_folder)
    mkdir(exp_path)
    print(f"Saving results to {exp_path}")

    # Save configuration info
    with open(join(exp_path, "config.json"), "w") as config_file:
        # Create a serializable configuration dictionary
        serializable_config = {
            "model_class": clsstr(model_cls),
            "model_parameter": model_args,
            "use_eca": use_eca,
            "use_dilated": use_dilated,
            "dataloader_class": clsstr(data_loader_cls),
            "processor_class": clsstr(processor_cls),
            "processor_parameter": processor_args,
            "optimizer_class": clsstr(optim_cls),
            "optimizer_parameter": optim_args,
            "scheduler_class": clsstr(scheduler_cls),
            "scheduler_parameter": scheduler_args,
            "training_parameter": {
                "drafters_set_train": getattr(set_train, "drafters", []),
                "drafters_set_val": getattr(set_val, "drafters", []),
                "drafter_set_test": [],
                "db_path": db_path,
                "batch_size": batch_size,
                "epochs": epoch_count,
                "name": name,
                "loss_fn": str(loss_fn),
                "acc_fn": str(acc_fn),
                "vis_fn": str(vis_fn),
                "debug": debug,
                "device": device.type,
            },
        }

        # Add model path if specified
        if model_path:
            serializable_config["model_path"] = model_path

        json.dump(serializable_config, config_file, indent=2)

    # Initialize model
    print("Initializing model...")
    model = model_cls(**model_args)

    # Apply ECA if requested and model is FasterRCNN
    if use_eca and isinstance(model, FasterRCNN):
        attach_eca(model, kernel_size=K_SIZE)
        print("ECA successfully attached to model")

    # Apply dilated convolutions if requested and model is FasterRCNN
    if use_dilated and isinstance(model, FasterRCNN):
        attach_dilated_convs(model, dilation_rates=[1, 2, 4])
        print("Dilated convolutions successfully attached to model")

    model.to(device)

    # Load pretrained model if specified
    if model_path:
        print(f"Loading pretrained model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    # Initialize optimizer and scheduler
    optimizer = optim_cls(model.parameters(), **optim_args)
    scheduler = scheduler_cls(optimizer, **scheduler_args)

    # Display model summary with low verbosity to save memory
    if debug:
        try:
            print(summary(model, verbose=1))
        except Exception as e:
            print(f"Model summary failed: {e}")

    # Track metrics
    curve_train_loss = []
    curve_val_loss = []
    curve_train_acc = []
    curve_val_acc = []
    best_val_acc = 0.0

    # Record training start time
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    print(
        f"Training started at: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Training loop
    epoch_times = []
    for epoch_nbr in range(epoch_count):
        epoch_start_time = time.time()
        print(f"\n==> Epoch {epoch_nbr} / {epoch_count} <==")

        # Force garbage collection between epochs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log_memory_stats(device)

        # Training phase
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
            exp_path=exp_path,
            vis_fn=vis_fn,
            debug=debug,
        )

        # Validation phase
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
            exp_path=exp_path,
            vis_fn=vis_fn,
            debug=debug,
        )

        # Check for NaN in training loss and terminate if found
        if torch.isnan(torch.tensor(train_loss)) or torch.isinf(
            torch.tensor(train_loss)
        ):
            print(
                f"\nERROR: Training loss is NaN or Inf at epoch {epoch_nbr}: {train_loss}"
            )
            print("Training terminated to prevent further corruption.")
            break

        # Record metrics
        curve_train_loss.append(train_loss)
        curve_val_loss.append(val_loss)
        curve_train_acc.append(train_acc)
        curve_val_acc.append(val_acc)

        # Plot learning curves
        plot_curve(
            {
                "Train Loss": curve_train_loss,
                "Val Loss": curve_val_loss,
                "Train Acc": curve_train_acc,
                "Val Acc": curve_val_acc,
            },
            epoch_count,
            exp_path,
            percent=True,
            fasterrcnn=True,
        )

        # Save current model
        torch.save(model.state_dict(), join(exp_path, "model_last.pt"))

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), join(exp_path, "model_best.pt"))
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

            # Copy eval results to best folder
            if vis_fn is not None:
                if not isdir(join(exp_path, "eval-best")):
                    mkdir(join(exp_path, "eval-best"))
                run(
                    f"cp {join(exp_path, 'eval', '*')} {join(exp_path, 'eval-best')}",
                    shell=True,
                )

        # Save metrics to file
        # with open(join(exp_path, "metrics.json"), "w") as f:
        #     metrics = {
        #         "train_loss": curve_train_loss,
        #         "val_loss": curve_val_loss,
        #         "train_acc": curve_train_acc,
        #         "val_acc": curve_val_acc,
        #         "best_val_acc": best_val_acc,
        #         "current_epoch": epoch_nbr,
        #     }
        #     json.dump(metrics, f, indent=4)

        # Record epoch completion time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch_nbr} completed in {epoch_duration:.2f} seconds")

    # Final save
    torch.save(model.state_dict(), join(exp_path, "model_final.pt"))

    # Save duration of epochs and whole training
    save_training_time(
        training_start_time, training_start_datetime, epoch_count, epoch_times, exp_path
    )


def main():
    """Parse command line arguments and start finetuning"""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m extraction.src.core.finetuning <config_file> [model_path]"
        )
        sys.exit(1)
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load the config file
    config_file = sys.argv[1]
    with open(config_file) as config_data:
        config = json.loads(config_data.read())

    # Load classes from config
    model_cls = class_from_package(config["model_class"])
    model_args = config["model_parameter"]
    optim_cls = class_from_package(config["optimizer_class"])
    optim_args = config["optimizer_parameter"]
    scheduler_cls = class_from_package(config["scheduler_class"])
    scheduler_args = config["scheduler_parameter"]
    data_loader_cls = class_from_package(config["dataloader_class"])
    processor_cls = class_from_package(config["processor_class"])
    processor_args = config["processor_parameter"]

    # Extract ECA setting (default to False if not specified)
    use_eca = config.get("use_eca", False)
    print(f"ECA (Efficient Channel Attention) enabled: {use_eca}")

    # Extract dilated convolutions setting (default to False if not specified)
    use_dilated = config.get("use_dilated", False)
    print(f"Dilated convolutions enabled: {use_dilated}")

    # Training parameters
    training_params = config["training_parameter"]
    loss_fn = class_from_package(training_params["loss_fn"])()
    acc_fn = class_from_package(training_params["acc_fn"])
    print(f"Using {acc_fn=}")

    print("== MEMORY BEFORE LOADING DATA ==")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_memory_stats(device)

    # Load data using the same approach as in training.py
    print("Loading Train set")
    set_train = data_loader_cls(
        drafters=training_params["drafters_set_train"],
        db_path=training_params["db_path"],
        augment=True,
        processor_cls=processor_cls,
        processor_args=processor_args,
        debug=training_params["debug"],
        caching=False,
    )

    print("Loading Validation set")
    set_val = data_loader_cls(
        drafters=training_params["drafters_set_val"],
        db_path=training_params["db_path"],
        augment=False,
        processor_cls=processor_cls,
        processor_args=processor_args,
        debug=training_params["debug"],
        caching=False,
    )

    # Create output directory if it doesn't exist
    save_path = join("extraction", "model")
    makedirs(save_path, exist_ok=True)

    # If model_path is not provided but exists in config, use that
    if model_path is None and "model_path" in config:
        model_path = config["model_path"]

    # Start finetuning
    finetune(
        model_cls=model_cls,
        model_args=model_args,
        optim_cls=optim_cls,
        optim_args=optim_args,
        scheduler_cls=scheduler_cls,
        scheduler_args=scheduler_args,
        data_loader_cls=data_loader_cls,
        processor_cls=processor_cls,
        processor_args=processor_args,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        set_train=set_train,
        set_val=set_val,
        db_path=training_params["db_path"],
        save_path=save_path,
        batch_size=training_params["batch_size"],
        epoch_count=training_params["epochs"],
        name=training_params["name"],
        vis_fn=(
            class_from_package(training_params["vis_fn"])
            if "vis_fn" in training_params
            else None
        ),
        debug=training_params["debug"],
        model_path=model_path,
        use_eca=use_eca,
        use_dilated=use_dilated,
    )


if __name__ == "__main__":
    main()

"""
Example usage:

-- object detection
poetry run python -m extraction.src.core.finetuning extraction/config/od_enh/od.json

-- our best model
poetry run python -m extraction.src.core.finetuning od-symbol/extraction/config/od_enh/od_focal-giou.json
"""

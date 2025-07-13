"""sample_cghd.py: Sample CGHD dataset while preserving instance & segmentation integrity"""

# Next time you delete your file and remove it from git history there is a way to recover it from __pycache__:
# Decompiled with PyLingual (https://pylingual.io)

import argparse
import os

__author__ = "Vojtěch Sýkora"

ALL_DRAFTERS = list(range(-1, 31))


def parse_args():
    parser = argparse.ArgumentParser(description="Sample CGHD dataset")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/cghd_raw",
        help="Path to the CGHD dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/cghd_sample",
        help="Path to save the sampled images",
    )
    parser.add_argument(
        "--drafters",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Drafters to sample from",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples to extract",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Answer yes to all prompts"
    )
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level")
    return parser.parse_args()


def copy_file(src_file, dst_file, verbose=0):
    if verbose > 0:
        print(f"Copying {src_file} to {dst_file}")
    os.system(f"cp {src_file} {dst_file}")


def sample_cghd_instances_segmentation(
    input_path: str,
    output_path: str,
    subdirs=["instances", "segmentation"],
    max_samples=5,
    drafters=[],
    yes=False,
    verbose=0,
):
    """Sample a few instances and segmentation images from the CGHD dataset"""
    if not os.path.exists(output_path):
        print(f"Creating output path {output_path}")
        os.makedirs(output_path)
    elif not yes:
        response = input(
            f"Output path {output_path} already exists...Do you want to overwrite it? (y/n): "
        )
        if response.lower() != "y":
            print("Operation cancelled.")
            return

    for drafter in drafters:

        if verbose > 0:
            print(f"\ndrafter={drafter}")

        # for instances, segmentation
        filenames = {}
        for subdir in subdirs:
            drafter_path = os.path.join(input_path, f"drafter_{drafter}", subdir)
            if not os.path.exists(drafter_path):
                print(f"  Path {drafter_path} does not exist")
                continue

            # get max_samples files
            filenames[subdir] = []
            files = os.listdir(drafter_path)
            for file in files[:max_samples]:
                filenames[subdir].append(file)

        # get union of all subbdirs files
        all_files = set.union(
            *[
                set((file.split(".")[0] for file in filenames[subdir]))
                for subdir in subdirs
            ]
        )

        # check if instances refined exist in input path
        create_dirs = subdirs + ["annotations", "images"]
        if "instances" in create_dirs:
            if os.path.exists(
                os.path.join(input_path, f"drafter_{drafter}", "instances_refined")
            ):
                create_dirs.append("instances_refined")

        # create output directories
        for subdir in create_dirs:
            outdir = os.path.join(output_path, f"drafter_{drafter}", subdir)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        # copy files in instances, segmentation
        for subdir, files in filenames.items():
            for file in files:
                src_file = os.path.join(input_path, f"drafter_{drafter}", subdir, file)
                dst_file = os.path.join(output_path, f"drafter_{drafter}", subdir, file)
                copy_file(src_file, dst_file, verbose=verbose)

        # copy instances_refined
        if "instances" in create_dirs:
            if os.path.exists(
                os.path.join(input_path, f"drafter_{drafter}", "instances_refined")
            ):
                for file in filenames["instances"]:
                    src_file = os.path.join(
                        input_path, f"drafter_{drafter}", "instances_refined", file
                    )
                    dst_file = os.path.join(
                        output_path, f"drafter_{drafter}", "instances_refined", file
                    )
                    copy_file(src_file, dst_file, verbose=verbose)

        # copy corresponding annotations, images
        for file in all_files:
            for subdir in ["annotations", "images"]:
                src_files = [
                    f
                    for f in os.listdir(
                        os.path.join(input_path, f"drafter_{drafter}", subdir)
                    )
                    if f.startswith(file)
                ]
                for src_file in src_files:
                    dst_file = os.path.join(
                        output_path, f"drafter_{drafter}", subdir, src_file
                    )
                    src_file = os.path.join(
                        input_path, f"drafter_{drafter}", subdir, src_file
                    )
                    copy_file(src_file, dst_file, verbose=verbose)
        print(f"Finished sampling drafter {drafter}")


if __name__ == "__main__":
    args = parse_args()
    print(f"args={args!r}")

    sample_cghd_instances_segmentation(
        args.input_path,
        args.output_path,
        drafters=args.drafters,
        max_samples=args.max_samples,
        yes=args.yes,
        verbose=args.verbose,
    )

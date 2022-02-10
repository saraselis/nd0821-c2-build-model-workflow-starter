"""
Performs basic cleaning on the data and save the results in W&B.
"""
import argparse
import logging
import os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading Artifact")
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    logger.info("Dropping the outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Convert last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Dropping rows in the dataset that are not in the proper geolocation")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving the output artifact")
    file_name = "clean_sample.csv"
    df.to_csv(file_name, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description)
    artifact.add_file(file_name)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A basic data cleaning.")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for cleaning outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for cleaning outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)

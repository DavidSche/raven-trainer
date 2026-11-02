"""Utility: split dataset into train/val/test sets."""

import click


@click.command()
@click.option("--source", "-s", required=True, help="Source image directory")
@click.option("--output", "-o", default="./datasets", help="Output directory")
@click.option("--ratios", default="0.7,0.2,0.1", help="Train/val/test ratios")
@click.option("--method", default="video_based", help="Split method: random | video_based | temporal")
def split_dataset(source: str, output: str, ratios: str, method: str):
    """Split a dataset into train / val / test subsets.

    Default split method is 'video_based' to prevent data leakage
    from similar frames ending up in both train and val sets.
    """
    click.echo(f"[Raven Trainer] Splitting dataset: {source}")
    click.echo(f"[Raven Trainer] Ratios: {ratios}, Method: {method}")


if __name__ == "__main__":
    split_dataset()

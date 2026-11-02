"""Utility: compute dataset statistics."""

import click


@click.command()
@click.option("--dataset-yaml", required=True, help="Path to dataset.yaml")
@click.option("--output", "-o", default="outputs/statistics", help="Output directory")
def statistics(dataset_yaml: str, output: str):
    """Compute dataset statistics:
    - Class distribution
    - Bounding box size distribution
    - Aspect ratio distribution
    - Occlusion statistics (if annotated)
    """
    click.echo(f"[Raven Trainer] Computing statistics: {dataset_yaml}")
    click.echo(f"[Raven Trainer] Output: {output}")


if __name__ == "__main__":
    statistics()

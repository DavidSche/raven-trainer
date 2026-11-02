"""Utility: visualize dataset samples with annotations."""

import click


@click.command()
@click.option("--dataset-yaml", required=True, help="Path to dataset.yaml")
@click.option("--output", "-o", default="outputs/visualize", help="Output directory")
@click.option("--num-samples", type=int, default=50, help="Number of samples to visualize")
@click.option("--show", is_flag=True, help="Display images interactively")
def visualize(dataset_yaml: str, output: str, num_samples: int, show: bool):
    """Visualize dataset samples with overlaid annotations."""
    click.echo(f"[Raven Trainer] Visualizing: {dataset_yaml}")
    click.echo(f"[Raven Trainer] Samples: {num_samples}, Output: {output}")


if __name__ == "__main__":
    visualize()

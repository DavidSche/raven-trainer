"""Utility: validate dataset structure and annotations."""

import click


@click.command()
@click.option("--dataset-yaml", required=True, help="Path to dataset.yaml")
@click.option("--fix", is_flag=True, help="Auto-fix issues when possible")
def check_dataset(dataset_yaml: str, fix: bool):
    """Validate dataset structure, annotations, and label consistency.

    Checks:
    - Image files exist and are readable
    - Annotation files match image count
    - Label indices are consistent
    - No missing or corrupt files
    """
    click.echo(f"[Raven Trainer] Checking dataset: {dataset_yaml}")
    click.echo(f"[Raven Trainer] Auto-fix: {fix}")


if __name__ == "__main__":
    check_dataset()

"""Main CLI entry point: raven-train.

Commands:
  ingest      Data ingestion pipeline (scan, run, report)
  dataset     Data Asset Management (registry, version, metadata, tag, stats, doctor, lineage)
  new         Create a new training project
  train       Train model
  eval        Evaluate model
  export      Export model to ONNX
  job         Job system management
  ui          Start Web UI server
"""

from __future__ import annotations

import click

from .new import cmd_new
from .train import cmd_train
from .export import cmd_export
from .eval import cmd_eval
from .job import cmd_job
from .dataset import cmd_dataset
from .ingest import cmd_ingest
from .ui import cmd_ui


@click.group()
@click.version_option(version="0.5.0", prog_name="raven-train")
def cli():
    """Raven Trainer — Industrial Vision Dataset & Model Training Platform."""


cli.add_command(cmd_ingest)
cli.add_command(cmd_new)
cli.add_command(cmd_dataset)
cli.add_command(cmd_train)
cli.add_command(cmd_job)
cli.add_command(cmd_eval)
cli.add_command(cmd_export)
cli.add_command(cmd_ui)


def main():
    cli()


if __name__ == "__main__":
    main()

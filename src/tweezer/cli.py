import sys
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

from .pipeline import Pipeline, PRESETS
from .modules import REGISTRY
from . import io as audio_io

console = Console()


def _show_module_list():
    table = Table(
        title="[bold cyan]tweezer modules[/]",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
    )
    table.add_column("name", style="bold yellow", no_wrap=True)
    table.add_column("description", style="dim")
    table.add_column("parameters", style="cyan")

    for name, cls in sorted(REGISTRY.items()):
        param_lines = []
        for pname, spec in cls.PARAMS.items():
            range_str = ""
            if spec.range is not None:
                if spec.type in (int, float):
                    range_str = f" [{spec.range[0]}–{spec.range[1]}]"
                else:
                    range_str = f" {spec.range}"
            param_lines.append(
                f"[bold]{pname}[/]={spec.default!r}{range_str}\n  [dim]{spec.description}[/]"
            )
        table.add_row(name, cls.DESCRIPTION, "\n".join(param_lines))

    console.print(table)


def _show_preset_list():
    table = Table(
        title="[bold cyan]tweezer presets[/]",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("preset", style="bold yellow", no_wrap=True)
    table.add_column("chain", style="dim")

    for name, chain in sorted(PRESETS.items()):
        parts = []
        for entry in chain:
            entry = dict(entry)
            mod = entry.pop("module")
            params = ", ".join(f"{k}={v}" for k, v in entry.items())
            parts.append(f"{mod}({params})" if params else mod)
        table.add_row(name, " → ".join(parts))

    console.print(table)


def _show_chain(pipeline: Pipeline):
    if not pipeline.modules:
        console.print("[yellow]Empty chain[/]")
        return
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("module", style="bold yellow")
    table.add_column("parameters", style="cyan")

    for i, mod in enumerate(pipeline.modules, 1):
        param_str = "  ".join(f"{k}={v}" for k, v in mod.params.items())
        table.add_row(str(i), mod.NAME, param_str)

    console.print(table)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_file", required=False, default=None)
@click.argument("module_args", nargs=-1)
@click.option("--list", "list_modules", is_flag=True, help="List all modules and parameters")
@click.option("--presets", "list_presets", is_flag=True, help="List all presets")
@click.option("--preset", type=str, default=None, help="Use a named preset chain")
@click.option(
    "--chain-file",
    type=click.Path(exists=True),
    default=None,
    help="Load chain from JSON file",
)
@click.option("--dry-run", is_flag=True, help="Show chain without processing")
@click.option("--mono", is_flag=True, help="Mix to mono before processing")
@click.option("--save-chain", type=click.Path(), default=None, help="Save chain to JSON file")
def main(
    input_file,
    module_args,
    list_modules,
    list_presets,
    preset,
    chain_file,
    dry_run,
    mono,
    save_chain,
):
    """
    Retrograde DSP processor — hardware failure mode emulator.

    \b
    Process a file through a chain of modules:
      tweezer input.wav ghost:bits=3 brr:filter_mode=3

    \b
    Use a preset:
      tweezer --preset sp1200 input.wav

    \b
    Load a JSON chain file:
      tweezer --chain-file chain.json input.wav

    \b
    Output is written as a sibling file: input.tweezer-01.wav
    """
    if list_modules:
        _show_module_list()
        return

    if list_presets:
        _show_preset_list()
        return

    if not input_file:
        click.echo(click.get_current_context().get_help())
        return

    # Build pipeline
    try:
        if chain_file:
            pipeline = Pipeline.from_json(chain_file)
        elif preset:
            pipeline = Pipeline.from_preset(preset)
        elif module_args:
            pipeline = Pipeline.from_args(list(module_args))
        else:
            console.print("[red]No modules, preset, or chain file specified.[/]")
            console.print("Run [bold]tweezer --list[/] to see available modules.")
            console.print("Run [bold]tweezer --presets[/] to see available presets.")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    console.print()
    _show_chain(pipeline)
    console.print()

    if save_chain:
        import json
        with open(save_chain, "w") as f:
            json.dump(pipeline.to_json_dict(), f, indent=2)
        console.print(f"[green]Chain saved to[/] {save_chain}")

    if dry_run:
        return

    # Determine output path
    output_path = audio_io.next_output_path(input_file)

    # Read audio
    try:
        audio, sr = audio_io.read_audio(input_file, mono=mono)
    except Exception as e:
        console.print(f"[red]Could not read[/] {input_file}: {e}")
        sys.exit(1)

    n_modules = len(pipeline.modules)
    n_channels = 1 if audio.ndim == 1 else audio.shape[1]
    duration = audio.shape[0] / sr

    console.print(
        f"[dim]{input_file}[/]  "
        f"[cyan]{sr}Hz[/]  "
        f"[cyan]{n_channels}ch[/]  "
        f"[cyan]{duration:.1f}s[/]  "
        f"→  [green]{output_path}[/]"
    )

    # Process with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[dim]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("processing", total=n_modules)

        def callback(i: int, mod):
            progress.update(task, completed=i, description=mod.NAME)

        try:
            result = audio_io.process_stereo(audio, sr, pipeline, callback)
            progress.update(task, completed=n_modules, description="done")
        except Exception as e:
            console.print(f"[red]Processing error:[/] {e}")
            raise

    # Write output
    try:
        audio_io.write_audio(output_path, result, sr)
    except Exception as e:
        console.print(f"[red]Could not write[/] {output_path}: {e}")
        sys.exit(1)

    console.print(f"[green]✓[/] {output_path}")

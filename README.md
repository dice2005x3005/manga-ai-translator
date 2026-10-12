# Manga AI Translator

A desktop app, in development, that translates manga pages from Japanese to English with AI
models that run entirely on your own computer. Translation and image quality come first; speed
comes second.

> **Status:** foundations only. The app can inspect your hardware, create translation projects
> from a folder of pages and manage model downloads. It cannot translate pages yet.

## Requirements

- Windows 10 or 11
- [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- 16 GB of RAM recommended
- Optional: an NVIDIA GPU with a recent driver. The app is tuned for a GTX 1660 (6 GB VRAM),
  and it also runs on the CPU, more slowly.

uv installs the right Python version (3.12) for the project. It does not touch any Python
already installed on the system.

## Installation

```bash
git clone https://github.com/dice2005x3005/manga-ai-translator.git
cd manga-ai-translator
```

Then install the dependencies with the PyTorch build that matches your machine:

```bash
uv sync --extra cuda   # NVIDIA GPU
uv sync --extra cpu    # no NVIDIA GPU
```

Check that everything is detected correctly:

```bash
uv run mangat doctor
```

## Commands

Run any command with `uv run mangat <command>`. Add `--help` to any command for all its options.

| Command | What it does |
|---|---|
| `doctor` | Reports OS, CPU, RAM, NVIDIA GPU, driver and PyTorch build, then recommends a device, precision and quality profile. |
| `doctor --test-gpu` | Also checks that the GPU computes correctly in fp32 and fp16. GTX 16xx cards are known to fail in fp16. |
| `doctor --json` | Prints the whole report as JSON, handy for sharing when asking for help. |
| `project new <folder>` | Creates a project with one page per image (JPG, PNG, WEBP), in natural order (`p2` before `p10`). |
| `project info <project>` | Shows the pages, their processing stages, and whether any original image is missing or changed (`--verify` hashes them). |
| `models list` | Shows the model catalog and which models are downloaded. |
| `models download <id>... \| --all` | Downloads models. Interrupted downloads resume, and every file is checked against its SHA-256. |
| `models verify [<id>...]` | Re-checks the SHA-256 of downloaded models. |
| `config show` | Prints the effective configuration and the file it came from. |
| `config init` | Writes a commented example config file to edit. |

Global options go before the command: `--config <file>`, `--log-level DEBUG|INFO|WARNING|ERROR`, `--version`.

The model catalog is still empty: models are added as each processing stage is built.

## Projects

`mangat project new D:\manga\chapter01` creates `D:\manga\chapter01.mangat\project.json` next to
the images folder:

- The original images are never modified. The project stores their paths relative to the project
  folder, so moving both folders together keeps the project working.
- Each page records its size, color mode and SHA-256 hash, plus the state of every processing stage.
- Text corrected by hand is protected: later automatic runs never overwrite it.

## Configuration

Settings are resolved in this order, later ones winning:

1. Built-in defaults.
2. A config file: the one passed with `--config`, else the one in the `MANGAT_CONFIG` environment
   variable, else `%LOCALAPPDATA%\MangaAITranslator\config.toml` if it exists.
3. Command line options.

| Setting | Default | Meaning |
|---|---|---|
| `paths.models_dir` | `%LOCALAPPDATA%\MangaAITranslator\models` | Where model weights are stored. They can take several GB. |
| `paths.logs_dir` | `%LOCALAPPDATA%\MangaAITranslator\Logs` | Where `mangat.log` is written. It always records DEBUG detail. |
| `runtime.device` | `auto` | `auto`, `cuda` or `cpu`. |
| `runtime.precision` | `fp32` | `fp32` or `fp16` for vision models. |
| `runtime.profile` | `quality` | `quality` or `balanced`. |
| `logging.level` | `INFO` | Console log level. |

Run `uv run mangat config init` to get a commented file with all of them.

## Development

```bash
uv sync --extra cpu
uv run pytest
uv run ruff check .
uv run ruff format .
```

The tests generate their own images, so they need no manga pages and no GPU. CI runs the same
checks on Windows for every push.

## Model weights and copyrighted content

- No model weights are stored in this repository. `mangat models download` fetches them from
  their original publishers, pinned to an exact version. Each model keeps its own license, shown
  by `mangat models list`.
- Manga pages are copyrighted. Never commit pages, projects or translated output. The
  `.gitignore` blocks image files, archives and project folders to help with this.

## License

[GPL-3.0-or-later](LICENSE)

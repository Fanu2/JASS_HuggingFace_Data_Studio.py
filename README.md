# JASS Hugging Face Data Studio v0.2

A local-first PySide6 research workspace for exploring Hugging Face datasets.

## What changed from v0.1

v0.2 turns the prototype into a more dataset-oriented studio.

### New

- Dataset Intelligence panel
- Hugging Face Dataset Viewer API split/config discovery
- Hub dataset size information when available
- Local Parquet statistics
- Non-null/null counts
- distinct-value counts
- detected column roles:
  - Audio
  - Image
  - Video
  - Text / categorical
  - Numeric
  - Boolean
  - Nested
- cleaner selected-record inspector
- binary audio payloads are hidden instead of dumping raw bytes
- embedded WAV/RIFF audio playback
- embedded image preview with automatic scaling
- image dimensions display
- save selected image
- play/stop controls
- audio seek slider
- duration display
- multimodal field detection
- retained bounded Parquet preview design

## First test dataset

```text
https://huggingface.co/datasets/dianavdavidson/Vaani-garo-mizo-majority-lg-English-no-transcript0
```

The studio is intentionally schema-driven. It does not hard-code the Vaani column names.

## Install

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install PySide6 pyarrow pandas huggingface_hub
```

## Run

```bash
python JASS_HuggingFace_Data_Studio.py
```

Paste a Hugging Face dataset URL or an `org/dataset` identifier.

## Architecture

```text
                         Hugging Face Hub
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
        Hub metadata                        Dataset Viewer API
              │                                   │
        repository files                  splits / size / stats
              │
              ▼
        selected Parquet
              │
        local HF cache
              │
      ┌───────┼─────────┐
      ▼       ▼         ▼
    Schema  Statistics Records
                        │
                ┌───────┼────────┐
                ▼       ▼        ▼
              Text    Audio    Image/Video
                        │
                        ▼
                  JASS Workspace
```

## Why Parquet?

Hugging Face documents Parquet as the performant format used by the Dataset Viewer. Parquet is columnar, supports row groups and metadata/statistics, and can reduce memory requirements through selective reads.

JASS v0.2 therefore keeps the local explorer bounded rather than reading an entire large dataset into memory.

## v0.2 limitations

This is still a desktop prototype.

Not yet included:

- full remote pagination across every row
- remote server-side search/filter UI
- SQL workspace
- DuckDB integration
- waveform rendering
- image viewer/gallery (basic embedded-image preview is now included)
- video playback
- multiple audio tracks
- dataset-card Markdown renderer
- authentication UI
- gated/private dataset login flow
- upload/push to Hugging Face

## Planned v0.3

### Dataset Viewer parity

- configurations/subsets
- split selector
- first-row preview from the Hub API
- remote search
- remote filtering
- remote pagination
- dataset statistics

### Multimodal Studio

- waveform display
- image viewer
- thumbnail gallery
- video player
- media metadata

### Analytics

- histograms
- categorical distributions
- duration distributions
- language distribution
- speaker distribution
- missing-data visualization

## Planned v0.4

### Large Dataset Engine

Introduce DuckDB for analytical querying.

Possible workflow:

```text
Hugging Face
     ↓
Parquet
     ↓
DuckDB
     ↓
SQL
     ↓
JASS Explorer
```

This is particularly attractive because Hugging Face documents DuckDB as one of the libraries suitable for working with Hub Parquet data.

## Hardware philosophy

Designed for ordinary desktop hardware.

No GPU is required.

No PyTorch is required.

The core stack is:

- Python
- PySide6
- huggingface_hub
- PyArrow
- pandas

## Relationship to the JASS Data Lab

- JASS JSON Explorer
- JASS Parquet Data Explorer
- JASS Parquet Image Explorer
- JASS GitHub Data Studio
- **JASS Hugging Face Data Studio**

The Hugging Face Studio is repository-oriented and dataset-aware. It is not intended to be merely another Parquet viewer.

## Data safety

The studio is an inspection tool.

It does not modify, overwrite, or push changes to the source Hugging Face repository.

Downloaded files are handled through the Hugging Face cache.

## Status

**Prototype — v0.2**

The foundation is now suitable for continued development toward a full local Hugging Face research workspace.

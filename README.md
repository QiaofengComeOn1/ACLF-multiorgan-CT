# Multi-organ CT analysis for incident ACLF

Python code for multi-organ CT prediction of incident acute-on-chronic liver failure and exploratory imaging clustering in patients with acutely decompensated cirrhosis.

The implementation covers five-channel Inception-v3 fine-tuning and feature extraction, organ and clinical feature fusion, stability selection, classifier fitting, probability calibration, prediction, interpretation, consensus clustering and survival analysis. Prediction configurations are ClinicalMP, Liver, Spleen, Muscle, LSMFusion and CLSMFusion.

## Installation

Use an activated Python 3.11 virtual environment. Install the core package from the repository root:

```bash
python -m pip install -e .
python -m aclf_ct --help
```

Optional dependencies support classifier benchmarking, image processing and attribution:

```bash
python -m pip install -e ".[benchmark,imaging,attribution]"
```

Dependencies and command-line entry points are defined in [pyproject.toml](pyproject.toml).

## Running the analyses

[WORKFLOW.md](docs/WORKFLOW.md) provides commands in analysis order. [INPUTS.md](docs/INPUTS.md) defines the tables, image manifests and fitted-model inputs.

The image workflow starts from packed five-slice organ images and masks. DICOM conversion, resampling, nnU-Net segmentation, quantitative CT measurements, the complete seven-backbone comparison and publication figure assembly are outside this repository. Patient-level data, images and trained weights are external inputs.

Each command writes to a new output directory. Cohort allocation and patient identifiers must remain consistent across inputs. Saved models, feature order and preprocessing must match. Loading pickle or joblib artifacts requires trusted local files and `--trust-model-files`.

## Repository structure

```text
src/
  aclf_ct/             Analysis modules and paper_profile.json
  common_pipeline.py   Shared analysis functions
tests/                Numerical and synthetic-input tests
docs/
  INPUTS.md            Input formats and model requirements
  WORKFLOW.md          Analysis commands
pyproject.toml         Dependencies and package configuration
README.md              Installation and usage
.gitignore             Data and temporary-file exclusions
```

## Tests

Run the included tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests use synthetic inputs and temporary files. Image-related tests require the optional dependencies. These tests assess implementation behaviour; they do not establish reproduction of the study results.

## Analysis version

Version 0.2.0 contains the current analysis configuration. It differs from the saved study runs in the ClinicalMP feature set, stability-selection procedure, image cropping, image-attribution target and organ-dependence analysis. The saved ClinicalMP run uses ALT, TBIL and INR; the distributed configuration specifies ALT, TBIL, INR, HGB, SCR and ALB. The manuscript's numerical results have not been reproduced with this revised implementation.

# Environment

Recorded 2026-11-01 at setup (session 1, Windows); session 2 (Linux cloud VM) below.

- OS: Windows 11 (Windows-11-10.0.26300-SP0), x86-64
- Python: 3.12.9 (tags/v3.12.9, MSC v.1942 64 bit (AMD64)), venv in `.venv/`
- git 2.45.2.windows.1
- No PyTorch (not needed for M1-M4; the optional CNN would add it later).

Setup (any OS):
```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # macOS / Linux
python -m pip install -e .                                # editable install of src/gridmon
```

## pip freeze (= requirements.txt)
```
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
fonttools==4.68.0
iniconfig==2.3.1
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.2
narwhals==2.28.0
numpy==2.5.4
packaging==26.3
pandas==3.1.0
pillow==13.0.0
pluggy==1.7.0
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==3.0.0
PyYAML==6.0.3
scikit-learn==1.9.1
scipy==1.18.1
six==1.17.0
threadpoolctl==3.7.0
tqdm==4.70.1
tzdata==2026.6
```
(`gridmon` itself is installed editable from this repo and is not listed.)

## Session 2 (PROMPT_02), 2026-11-01 — Linux cloud VM
- OS: Linux 6.18 x86-64 (4 vCPU, 15 GB RAM); Python 3.12.3 (`python3.12 -m venv .venv`).
- `pip install -r requirements.txt` installed every pin unchanged (colorama is pure Python and
  installs on Linux), so no pin was dropped or relaxed. numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1,
  pandas 3.1.0, matplotlib 3.11.2.
- Cross-platform reproducibility: M1/M2 reproduce exactly (val macro-F1 0.933, identical confusion;
  int16 inputs of all golden vectors bit-identical; float64 traces within 3e-14). The AA FIR redesign
  differs from the Windows file in the last bit of 3 taps (libm); make_dataset now keeps the committed
  file when equal to 1e-12 (D-038).
- Runtimes (this VM): full `run_all.py` (M1-M2) ~200 s (dataset 145 s, baseline 48 s, golden 1 s).

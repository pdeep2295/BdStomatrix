# Platform guide

## macOS Apple silicon

Use the tested MPS route:

```bash
python run_bdstomatrix.py --input "/path/to/images" --output-dir "/path/to/output" --device mps
```

## Ubuntu or Windows with NVIDIA CUDA

Install an NVIDIA driver and the CUDA-enabled PyTorch build appropriate for
the operating system and driver. Follow the official PyTorch installation
selector rather than assuming the generic `requirements.txt` installs the
desired CUDA runtime.

First create and activate the base environment:

```bash
conda env create --file environment.yaml
conda activate bdstomatrix
```

Then use the official PyTorch installation selector to replace the generic
`torch` and `torchvision` packages with the CUDA build compatible with the
target driver. Confirm CUDA before running BdStomatrix:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

```text
--device cuda --cuda-device 0
```

The pipeline validates CUDA availability and the requested zero-based GPU
index. It never silently falls back to CPU.

Windows PowerShell example:

```powershell
python "C:\path\to\BdStomatrix_final\run_bdstomatrix.py" --input "C:\path\to\images" --output-dir "C:\path\to\new_output" --device cuda --cuda-device 0
```

Ubuntu example:

```bash
python "/path/to/BdStomatrix_final/run_bdstomatrix.py" \
--input "/path/to/images" \
--output-dir "/path/to/new_output" \
--device cuda \
--cuda-device 0
```

## CPU fallback by explicit request

CPU execution is supported on macOS, Linux and Windows:

```text
--device cpu
```

It is expected to be slower for detection, classification and segmentation,
but trait extraction itself is predominantly CPU-bound.



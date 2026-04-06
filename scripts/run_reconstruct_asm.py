from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cgh_depth.analysis import _analysis_device, _load_exr_tensor, _normalize_image, reconstruct_asm


def _load_hologram_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix.lower() == ".exr":
        return _load_exr_tensor(path, device)

    image = plt.imread(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    image = image.astype(np.float32)
    if float(image.max()) > 1.0:
        image = image / 255.0
    return torch.from_numpy(image).float().to(device)


def _default_output_path(amp_path: Path, z_mm: float) -> Path:
    output_dir = ROOT / "results" / "asm_reconstruction"
    return output_dir / f"{amp_path.stem}_reconstruction_{z_mm:.1f}mm.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp-path", required=True, help="Amplitude image path (.exr, .png, etc.)")
    parser.add_argument("--phs-path", required=True, help="Phase image path (.exr, .png, etc.)")
    parser.add_argument(
        "--z-mm",
        type=float,
        default=10.0,
        help="Propagation distance in millimeters.",
    )
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--wavelength", type=float, default=638e-9)
    parser.add_argument(
        "--phase-normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpret phase input as normalized [0, 1]. Use --no-phase-normalized for radians.",
    )
    parser.add_argument(
        "--output-path",
        help="Output PNG path. Defaults to results/asm_reconstruction/<amp_stem>_reconstruction_<z>.png",
    )
    args = parser.parse_args()

    device = _analysis_device()
    amp_path = Path(args.amp_path)
    phs_path = Path(args.phs_path)
    amp = _load_hologram_tensor(amp_path, device)
    phs = _load_hologram_tensor(phs_path, device)

    if amp.shape != phs.shape:
        raise ValueError(f"Amplitude and phase shapes must match, got {amp.shape} and {phs.shape}.")

    reconstruction = reconstruct_asm(
        amp,
        phs,
        z=args.z_mm / 1000.0,
        pitch=args.pitch,
        wavelength=args.wavelength,
        phase_normalized=args.phase_normalized,
    )

    output_path = Path(args.output_path) if args.output_path else _default_output_path(amp_path, args.z_mm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, _normalize_image(reconstruction), cmap="gray")
    print(f"Saved ASM reconstruction: {output_path}")


if __name__ == "__main__":
    main()

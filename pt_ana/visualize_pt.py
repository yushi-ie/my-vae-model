import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a saved activation tensor (.pt) using PCA."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a .pt file or a directory that contains .pt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image file (only when --input is a single file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PCA plots will be stored when processing multiple files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan directories recursively for .pt files.",
    )
    parser.add_argument(
        "--sample-dim",
        type=int,
        default=1,
        help="Tensor dimension treated as samples (default: 1 for channel-first data).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=2,
        help="Number of PCA components (1 or 2) to visualize.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--x-lim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="Fix x-axis range (e.g., --x-lim -5 5).",
    )
    parser.add_argument(
        "--y-lim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Fix y-axis range (for PC1 or PC2).",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_pca.png")


def _gather_inputs(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {path}")
    pattern = "**/*.pt" if recursive else "*.pt"
    files = sorted(path.rglob("*.pt") if recursive else path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {path}")
    return files


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "tensor" in obj and torch.is_tensor(obj["tensor"]):
            tensor = obj["tensor"]
        else:
            for value in obj.values():
                if torch.is_tensor(value):
                    tensor = value
                    break
            else:
                raise ValueError("Dictionary does not contain a tensor entry.")
    elif torch.is_tensor(obj):
        tensor = obj
    else:
        tensor = torch.as_tensor(obj)
    return tensor.detach().cpu()


def prepare_matrix(tensor: torch.Tensor, sample_dim: int) -> torch.Tensor:
    dims = tensor.ndim
    if dims < 2:
        raise ValueError("Tensor must have at least 2 dimensions for PCA visualization.")

    if sample_dim < 0:
        sample_dim += dims
    if not 0 <= sample_dim < dims:
        raise ValueError(f"sample_dim {sample_dim} is out of bounds for tensor with {dims} dims.")

    perm = [sample_dim] + [d for d in range(dims) if d != sample_dim]
    permuted = tensor.permute(perm)
    samples = permuted.shape[0]
    matrix = permuted.reshape(samples, -1).to(torch.float32)
    return matrix


def compute_pca(matrix: torch.Tensor, components: int) -> torch.Tensor:
    components = max(1, min(components, matrix.size(0)))
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if centered.size(0) == 1:
        return centered
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    comps = min(components, s.size(0))
    return u[:, :comps] * s[:comps]


def plot_scores(
    scores: torch.Tensor,
    out_path: Path,
    title: str | None,
    width: float,
    height: float,
    x_lim: tuple[float, float] | None,
    y_lim: tuple[float, float] | None,
) -> None:
    dims = scores.size(1)
    plt.figure(figsize=(width, height))
    if dims == 1:
        values = scores[:, 0].numpy()
        plt.plot(values, marker="o")
        plt.xlabel("Sample Index")
        plt.ylabel("PC1")
        if y_lim:
            plt.ylim(y_lim)
    else:
        x = scores[:, 0].numpy()
        y = scores[:, 1].numpy()
        plt.scatter(x, y, c=range(len(x)), cmap="viridis", s=25)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(label="Sample Index")
        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)
    plt.title(title or "PCA of Activation Tensor")
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    inputs = _gather_inputs(args.input, args.recursive)

    multi_mode = len(inputs) > 1 or args.input.is_dir()

    results = []
    for input_path in inputs:
        tensor = load_tensor(input_path)
        matrix = prepare_matrix(tensor, args.sample_dim)
        scores = compute_pca(matrix, components=args.components)

        if multi_mode:
            out_dir = args.output_dir or args.output or input_path.parent
            out_path = out_dir / f"{input_path.stem}_pca.png"
            title = args.title or input_path.stem
        else:
            out_path = args.output or _default_output_path(input_path)
            title = args.title

        plot_scores(
            scores,
            out_path,
            title,
            args.fig_width,
            args.fig_height,
            tuple(args.x_lim) if args.x_lim else None,
            tuple(args.y_lim) if args.y_lim else None,
        )
        results.append(out_path)
        print(f"PCA visualization saved to {out_path}")

    if multi_mode:
        print(f"Processed {len(results)} tensors from {args.input}")


if __name__ == "__main__":
    main()

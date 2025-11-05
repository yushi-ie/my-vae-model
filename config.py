from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    z_dim: int = 20
    batch_size: int = 128
    test_batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    step_size: int = 15
    gamma: float = 0.1
    num_workers: int = 2
    download: bool = True
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    recon_filename: str = "reconstructions.png"
    sample_filename: str = "average_sample.png"

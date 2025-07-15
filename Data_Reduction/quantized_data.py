import numpy as np
import csv
import os
import zipfile
from typing import List
import tempfile


def write_quantized_csv(
    time_series_list: List[np.ndarray], output_dir: str, num_levels: int = 256
):
    """
    Writes:
    - metadata.csv: length, min, and max for each time series.     --> could be columns in quantized.csv as well but this is more structured.
    - quantized.csv: 8-bit quantized values of the time series.   --> each for its own min max values
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "metadata.csv"), "w", newline=""
    ) as meta_file, open(
        os.path.join(output_dir, "quantized.csv"), "w", newline=""
    ) as data_file:

        meta_writer = csv.writer(meta_file)
        data_writer = csv.writer(data_file)

        meta_writer.writerow(["length", "min", "max"])

        for series in time_series_list:
            length = len(series)
            min_val = float(series.min())
            max_val = float(series.max())
            scale = (max_val - min_val) if max_val != min_val else 1.0

            q = np.clip(
                ((series - min_val) / scale * (num_levels - 1)).round(),
                0,
                num_levels - 1,
            ).astype(int)

            meta_writer.writerow([length, f"{min_val:.6g}", f"{max_val:.6g}"])
            data_writer.writerow(q.tolist())


def zip_csv_files(output_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(output_dir, "metadata.csv"), arcname="metadata.csv")
        zf.write(os.path.join(output_dir, "quantized.csv"), arcname="quantized.csv")


def write_quantized_zip(
    time_series_list: List[np.ndarray],
    output_dir: str,
    file_name: str,
    num_levels: int = 256,
):
    signals = [np.array(sig, dtype=np.float32) for sig in time_series_list]
    write_quantized_csv(signals, output_dir, num_levels)
    zip_path = os.path.join(output_dir, file_name)
    zip_csv_files(output_dir, zip_path)
    os.remove(os.path.join(output_dir, "metadata.csv"))
    os.remove(os.path.join(output_dir, "quantized.csv"))


def read_quantized_zip(zip_path: str, num_levels: int = 256) -> List[np.ndarray]:

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        meta_path = os.path.join(tmpdir, "metadata.csv")
        data_path = os.path.join(tmpdir, "quantized.csv")
        results = []

        with open(meta_path, newline="") as meta_file, open(
            data_path, newline=""
        ) as data_file:
            meta_reader = csv.DictReader(meta_file)
            data_reader = csv.reader(data_file)

            for meta_row, q_row in zip(meta_reader, data_reader):
                length = int(meta_row["length"])
                min_val = float(meta_row["min"])
                max_val = float(meta_row["max"])
                scale = (max_val - min_val) if max_val != min_val else 1.0

                q = np.array(list(map(int, q_row[:length])), dtype=np.uint8)
                recon = min_val + (q.astype(np.float32) / (num_levels - 1)) * scale
                results.append(recon)

        return results



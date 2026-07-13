"""Resume-download the iNat2021 train_mini archive for E-VQA.

The archive is large and the S3 connection can be flaky. This downloader splits
the file into byte ranges and resumes each part by appending from the last
downloaded byte. After all parts are complete, it combines them into
``raw/train_mini.tar.gz``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import threading
import time
from pathlib import Path

import requests


DEFAULT_SOURCE_ROOT = Path("/media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA")
DEFAULT_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"
)
DEFAULT_SIZE = 44_636_137_542


def expected_range(index: int, *, size: int, parts: int) -> tuple[int, int]:
    start = size * index // parts
    end = size * (index + 1) // parts - 1
    if index == parts - 1:
        end = size - 1
    return start, end


def expected_size(index: int, *, size: int, parts: int) -> int:
    start, end = expected_range(index, size=size, parts=parts)
    return end - start + 1


def part_size(part_dir: Path, index: int) -> int:
    path = part_dir / f"part_{index:02d}"
    return path.stat().st_size if path.exists() else 0


def all_sizes(part_dir: Path, parts: int) -> list[int]:
    return [part_size(part_dir, index) for index in range(parts)]


def download_part(
    index: int,
    *,
    url: str,
    size: int,
    parts: int,
    part_dir: Path,
    lock: threading.Lock,
) -> str:
    start, end = expected_range(index, size=size, parts=parts)
    expected = end - start + 1
    path = part_dir / f"part_{index:02d}"
    session = requests.Session()
    attempts = 0

    while True:
        existing = part_size(part_dir, index)
        if existing == expected:
            return f"part_{index:02d} complete"
        if existing > expected:
            path.unlink()
            existing = 0

        attempts += 1
        headers = {"Range": f"bytes={start + existing}-{end}"}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}; expected 206")
                with path.open("ab" if existing else "wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        except Exception as exc:  # noqa: BLE001 - network retries are expected.
            with lock:
                print(
                    f"part_{index:02d} retry {attempts}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            time.sleep(min(60, 5 + attempts))


def combine_parts(*, part_dir: Path, output: Path, size: int, parts: int) -> None:
    for index in range(parts):
        actual = part_size(part_dir, index)
        expected = expected_size(index, size=size, parts=parts)
        print(f"part_{index:02d}: {actual}/{expected}", flush=True)
        if actual != expected:
            raise RuntimeError(f"bad size for part_{index:02d}")

    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as writer:
        for index in range(parts):
            with (part_dir / f"part_{index:02d}").open("rb") as reader:
                while True:
                    chunk = reader.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    writer.write(chunk)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"combined size mismatch: {tmp.stat().st_size} != {size}")
    tmp.replace(output)
    print(f"combined {output} ({output.stat().st_size} bytes)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-seconds", type=int, default=120)
    args = parser.parse_args(argv)

    raw_dir = args.source_root / "raw"
    part_dir = raw_dir / "train_mini_parts"
    output = raw_dir / "train_mini.tar.gz"
    part_dir.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    lock = threading.Lock()

    def reporter() -> None:
        while not stop.wait(args.report_seconds):
            sizes = all_sizes(part_dir, args.parts)
            total = sum(sizes)
            completed = sum(
                1
                for index, value in enumerate(sizes)
                if value == expected_size(index, size=args.size, parts=args.parts)
            )
            print(
                f"progress {total / 1024**3:.2f}/{args.size / 1024**3:.2f} GiB "
                f"({total / args.size:.2%}); completed_parts={completed}/{args.parts}; "
                f"min={min(sizes) / 1024**2:.1f}MiB "
                f"max={max(sizes) / 1024**2:.1f}MiB",
                flush=True,
            )

    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()

    order = sorted(
        range(args.parts),
        key=lambda index: part_size(part_dir, index)
        / expected_size(index, size=args.size, parts=args.parts),
    )
    print(f"download order: {order}", flush=True)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        submitted = [
            executor.submit(
                download_part,
                index,
                url=args.url,
                size=args.size,
                parts=args.parts,
                part_dir=part_dir,
                lock=lock,
            )
            for index in order
        ]
        for future in futures.as_completed(submitted):
            print(future.result(), flush=True)

    stop.set()
    combine_parts(part_dir=part_dir, output=output, size=args.size, parts=args.parts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

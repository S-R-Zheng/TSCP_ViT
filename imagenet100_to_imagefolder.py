import os
import shutil
import zipfile
import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jpeg", ".JPEG", ".JPG", ".PNG"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in {e.lower() for e in IMG_EXTS})


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool = False) -> None:
    if overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)
    safe_mkdir(extract_dir)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extract_dir))


def find_dataset_root(extract_dir: Path) -> Path:

    required_train = {f"train.X{i}" for i in range(1, 5)}
    required_val = "val.X"


    for train_x1 in extract_dir.rglob("train.X1"):
        if not train_x1.is_dir():
            continue
        parent = train_x1.parent
        siblings = {p.name for p in parent.iterdir() if p.is_dir()}
        if required_train.issubset(siblings) and required_val in siblings:
            return parent

    raise FileNotFoundError(
    f"No directory was found under the extraction path '{extract_dir}' that contains both "
    "train.X1~train.X4 and val.X. "
    "Please verify that the ZIP file follows the Kaggle ImageNet100 directory structure."
    )


def transfer_file(src: Path, dst: Path, method: str) -> None:

    if method == "move":
        shutil.move(str(src), str(dst))
    elif method == "copy":
        shutil.copy2(str(src), str(dst))
    elif method == "hardlink":
        os.link(str(src), str(dst))
    elif method == "symlink":
        rel = os.path.relpath(str(src), start=str(dst.parent))
        os.symlink(rel, str(dst))
    else:
        raise ValueError(f"Unknown method: {method}")


def unique_dst_path(dst_dir: Path, filename: str, tag: str) -> Path:
    dst = dst_dir / filename
    if not dst.exists():
        return dst

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    cand = dst_dir / f"{stem}_{tag}{suffix}"
    if not cand.exists():
        return cand

    k = 1
    while True:
        cand = dst_dir / f"{stem}_{tag}_{k}{suffix}"
        if not cand.exists():
            return cand
        k += 1


def convert_split(src_roots, out_root: Path, method: str, split_name: str) -> tuple[int, int]:
    safe_mkdir(out_root)
    classes_seen = set()
    num_images = 0

    for src_root in src_roots:
        shard_tag = src_root.name
        if not src_root.exists():
            raise FileNotFoundError(f"Missing directory: {src_root}")

        for cls_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
            cls_name = cls_dir.name
            classes_seen.add(cls_name)

            dst_cls_dir = out_root / cls_name
            safe_mkdir(dst_cls_dir)

            for f in cls_dir.iterdir():
                if not is_image_file(f):
                    continue
                dst_path = unique_dst_path(dst_cls_dir, f.name, shard_tag)
                transfer_file(f, dst_path, method)
                num_images += 1

    print(f"[{split_name}] classes={len(classes_seen)} images={num_images}")
    return len(classes_seen), num_images


def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle ImageNet100 archive.zip to ImageFolder structure.")
    parser.add_argument("--zip_path", type=str, default="./archive.zip",
                        help="Path to archive.zip")
    parser.add_argument("--out_root", type=str, default="./imagenet-100",
                        help="Output root directory (will create train/ and val/ under it)")
    parser.add_argument("--work_dir", type=str, default="./_imagenet100_extracted",
                        help="Temporary directory to extract zip contents")
    parser.add_argument("--method", type=str, default="move",
                        choices=["move", "copy", "hardlink", "symlink"],
                        help="How to place files into output: move/copy/hardlink/symlink")
    parser.add_argument("--overwrite_extract", action="store_true",
                        help="If set, delete work_dir before extracting")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).resolve()
    out_root = Path(args.out_root).resolve()
    work_dir = Path(args.work_dir).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    print(f"Zip:      {zip_path}")
    print(f"Work dir: {work_dir}")
    print(f"Output:   {out_root}")
    print(f"Method:   {args.method}")

    print("1) Extracting zip ...")
    extract_zip(zip_path, work_dir, overwrite=args.overwrite_extract)

    print("2) Locating dataset root ...")
    ds_root = find_dataset_root(work_dir)
    print(f"Dataset root found: {ds_root}")

    train_shards = [ds_root / f"train.X{i}" for i in range(1, 5)]
    val_src = ds_root / "val.X"

    out_train = out_root / "train"
    out_val = out_root / "val"

    print("3) Converting train shards -> train/ ...")
    convert_split(train_shards, out_train, args.method, "train")

    print("4) Converting val.X -> val/ ...")
    convert_split([val_src], out_val, args.method, "val")

    print("Done.")
    print(f"Now you can load:")
    print(f"  ImageFolder('{out_train}')")
    print(f"  ImageFolder('{out_val}')")


if __name__ == "__main__":
    main()

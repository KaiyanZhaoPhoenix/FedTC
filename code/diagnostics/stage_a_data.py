"""Prepare the fixed CIFAR-100-LT feature benchmark used by the Stage A pilot."""
import argparse
import hashlib
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import squeezenet1_1

ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
ENCODER_SHA = "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b"


@torch.inference_mode()
def prepare(folder):
    archive = folder / "cifar-100-python.tar.gz"
    weights = folder / "squeezenet1_1-b8a52dc0.pth"
    if hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5 or hashlib.sha256(weights.read_bytes()).hexdigest() != ENCODER_SHA:
        raise ValueError("CIFAR archive or encoder checksum mismatch")
    encoder = squeezenet1_1(weights=None)
    encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    encoder.eval()
    torch.set_num_threads(3)
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    std = torch.tensor([.229,.224,.225])[None,:,None,None]
    arrays = {}
    with tarfile.open(archive) as tar:
        for split in ("train","test"):
            record = pickle.load(tar.extractfile("cifar-100-python/"+split),encoding="bytes")
            labels = np.asarray(record[b"fine_labels"])
            indices = np.arange(len(labels))
            if split == "train":
                rng = np.random.default_rng(4091)
                indices = np.concatenate([rng.choice(np.flatnonzero(labels==c),round(500*10**(-c/99)),replace=False)
                                          for c in range(100)])
            images = record[b"data"].reshape(-1,3,32,32)[indices]
            batches = []
            for start in range(0,len(images),96):
                pixels = torch.from_numpy(images[start:start+96]).float()/255.
                pixels = F.interpolate(pixels,size=(64,64),mode="bilinear",align_corners=False)
                batches.append(encoder.features((pixels-mean)/std).mean(dim=(2,3)).numpy())
            arrays[split+"_x"] = np.concatenate(batches)
            arrays[split+"_y"] = labels[indices]
    path = folder/"cifar100_lt_features.npz"
    np.savez(path,**arrays)
    print(path, {k:v.shape for k,v in arrays.items()}, flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    prepare(parser.parse_args().folder)

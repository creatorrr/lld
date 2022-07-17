# llg/loader.py

import os

import h5py
import numpy as np
import PIL.Image as Image

script_dir = os.path.dirname(__file__)
datafile_path = os.path.join(script_dir, "../raw/LLD-logo.hdf5")

with h5py.File(datafile_path, "r") as throwaway:
    samples_count: int = len(throwaway["data"])


def gen_samples(
    labels: list[str] = ["data", "meta_data/names"], datafile_path: str = datafile_path
):

    # open hdf5 file
    with h5py.File(datafile_path, "r") as hdf5_file:
        count = len(hdf5_file["data"])

        i = 0
        while i < count:
            result = {}

            if "data" in labels:
                shape = hdf5_file["shapes"][i]
                images = hdf5_file["data"][i][:, : shape[1], : shape[2]]

                result["data"] = images.astype(np.uint8)

            for label in [l for l in labels if l != "data"]:
                result[label] = hdf5_file[label][i]

            yield result

            i += 1


if __name__ == "__main__":
    sample = next(gen_samples())
    name = sample["meta_data/names"]
    images = sample["data"]

    print(name)

    image_pil = Image.fromarray(images[2])
    image_pil.show()

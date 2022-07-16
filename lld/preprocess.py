# llg/preprocess.py

import argparse, asyncio, os
from itertools import islice

from aiostream import stream
import h5py as h5
import numpy as np
import pandas as pd
from tqdm.asyncio import trange

from .loader import gen_samples, samples_count
from .crawler import run

script_dir = os.path.dirname(__file__)
outfile_path = os.path.join(script_dir, "../data/lld-processed.h5")


async def gen_processor(batch_size: int, limit: int):
    count = min(limit, samples_count)
    batch_size = min(limit, batch_size)

    samples = gen_samples()
    steps = count // batch_size

    for step in trange(steps):
        batch = list(islice(samples, step * batch_size, (step + 1) * batch_size))

        urls = [f"http://{sample['meta_data/names'].decode()}.com" for sample in batch]
        descriptions = await run(urls, batch_size)

        for sample, description in zip(batch, descriptions):
            name = (sample["meta_data/names"].decode(),)
            images = sample["data"]

            data = (
                images,
                description,
                name,
            )

            yield data


async def preprocess(batch_size: int = 100, limit: int = samples_count + 1):

    columns = ["images", "description", "name"]

    processor = gen_processor(batch_size, limit)

    chunk_size = 10
    async with stream.chunks(processor, chunk_size).stream() as chunks:
        async for chunk in chunks:
            df_chunk = pd.DataFrame(chunk, columns=columns)
            df_chunk.to_hdf(
                outfile_path, "data", data_columns=columns, mode="a"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        help="Limit to total records processed",
        type=int,
        default=samples_count + 1,
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size",
        type=int,
        nargs="?",
        const=10_000,
        default=10_000,
    )

    args = parser.parse_args()

    asyncio.run(preprocess(batch_size=args.batch_size, limit=args.limit))

"""Dataset class for LLD dataset."""

import datasets as ds
import pandas as pd
from sklearn.model_selection import train_test_split


_HOMEPAGE = "https://huggingface.co/datasets/diwank/lld"
_LICENSE = "MIT"

_DESCRIPTION = """
Designing a logo for a new brand is a lengthy and tedious back-and-forth process between a designer and a client. In this paper we explore to what extent machine learning can solve the creative task of the designer. For this, we build a dataset -- LLD -- of 600k+ logos crawled from the world wide web. Training Generative Adversarial Networks (GANs) for logo synthesis on such multi-modal data is not straightforward and results in mode collapse for some state-of-the-art methods. We propose the use of synthetic labels obtained through clustering to disentangle and stabilize GAN training. We are able to generate a high diversity of plausible logos and we demonstrate latent space exploration techniques to ease the logo design task in an interactive manner. Moreover, we validate the proposed clustered GAN training on CIFAR 10, achieving state-of-the-art Inception scores when using synthetic labels obtained via clustering the features of an ImageNet classifier. GANs can cope with multi-modal data by means of synthetic labels achieved through clustering, and our results show the creative potential of such techniques for logo synthesis and manipulation.
"""

_CITATION = """
@misc{sage2017logodataset,
author={Sage, Alexander and Agustsson, Eirikur and Timofte, Radu and Van Gool, Luc},
title = {LLD - Large Logo Dataset - version 0.1},
year = {2017},
"""

_URL = "https://huggingface.co/datasets/diwank/lld/resolve/main/data/lld-processed.h5"


class LLD(ds.GeneratorBasedBuilder):
    """LLD Images dataset."""

    def _info(self):
        print("_info(self):")
        import pdb; pdb.set_trace()
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=ds.Features(
                {
                    "image": ds.Sequence(feature=ds.Image()),
                    "description": ds.Value("string"),
                }
            ),
            supervised_keys=("image", "description"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        print("_split_generators(self, dl_manager):")
        import pdb; pdb.set_trace()
        # Load dataframe
        use_local = os.environ.get("USE_LOCAL")
        archive_path = (
            "./data/lld-processed.h5" if use_local else dl_manager.download(_URL)
        )
        df = pd.read_hdf(archive_path)

        X = df.pop("description")
        y = df.pop("images")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "description": X_train,
                    "images": y_train,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={
                    "description": X_test,
                    "images": y_test,
                },
            ),
        ]

    def _generate_examples(self, description, images):
        print("_generate_examples(self, description, images):")
        import pdb; pdb.set_trace()
        """Generate images and description splits."""

        for i, (desc, imgs) in enumerate(zip(description.values, images.values)):
            for img in imgs:
                yield i, {
                    "image": {"bytes": img},
                    "description": desc,
                }

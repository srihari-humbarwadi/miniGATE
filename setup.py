import datetime

from setuptools import setup

base_version = "0.3"
version_str = f"{base_version}.{datetime.date.today().month}.{datetime.date.today().day}.{datetime.datetime.now().hour}"

setup(
    name="minigate",
    version=version_str,
    packages=[
        "minigate.learners",
        "minigate.model_blocks",
        "minigate.model_blocks.auto_builder_modules",
        "minigate.datasets",
        "minigate.datamodules",
        "minigate.datamodules.tf_hub",
        "minigate.models",
        "minigate.tasks",
        "minigate.base",
        "minigate.base.callbacks",
        "minigate.base.utils",
        "minigate.base.vendor",
        "minigate.configs",
        "minigate",
    ],
    url="https://github.com/AntreasAntoniou/GATE",
    license="GNU-v-3.0",
    author="Antreas Antoniou",
    author_email="a.antoniou@ed.ac.uk",
    description="Generalization After Transfer Evaluation on Modality, "
    "Task and Data Domain - A codebase enabling evaluation "
    "of architectures after transfer to a number of novel "
    "data domains, tasks and modalities",
)

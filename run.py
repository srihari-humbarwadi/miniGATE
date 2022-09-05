import os
import warnings

import dotenv
import hydra
import rich
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install
from rich.tree import Tree

from minigate.base.utils.loggers import get_logger
from minigate.base.utils.rank_zero_ops import extras

# load environment variables from `.env-` file if it exists
# recursively searches for `.env` in all folders starting from work dir

dotenv.load_dotenv(override=True, verbose=True)
install(show_locals=False, word_wrap=True, width=350)
log = get_logger(__name__)


from minigate.configs.config import collect_config_store

config_store = collect_config_store()


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from minigate.train_eval import train_eval

    os.environ["WANDB_PROGRAM"] = config.code_dir

    extras(config)
    if config.get("use_debug_entry_point"):
        from minigate.debug import debug

        return debug(config)
    return train_eval(config)


if __name__ == "__main__":
    main()

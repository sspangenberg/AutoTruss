# python Stage1/continuous_uct_3d.py --config $1 --run-id $2
# python Stage1/noise_input_permutation_format_transfer.py --config $1 --run-id $2
# python Stage2/main_3d.py --config $1 --run-id $2 --EmbeddingBackbone Transformer
# python apps/draw.py --config $1 --run-id $2 --draw-file PostResults/$1/$2/_best.txt

import sys

from configs.config import get_base_config, make_config
from stage_2 import main_3d


def main():
    parser = get_base_config()
    args = parser.parse_known_args(sys.argv[1:])[0]
    config = make_config(args.config)
    for k, v in config.get("base", {}).items():
        if f"--{k}" not in args:
            setattr(args, k, v)

    # continuous_uct_3d.main(args)
    # noise_input_permutation_format_transfer.main(args)
    main_3d.main(args)


if __name__ == "__main__":
    main()

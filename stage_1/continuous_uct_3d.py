import os

from algo.ucts import UCTs_init, UCTSearch
from truss_envs.reward import Envs_init, reward_fun
from utils.utils import readAlist, readFile, util_init


def main(args):
    if not os.path.exists("results_3d/"):
        os.mkdir("results_3d/")

    p, e = readFile(args.input_path)
    if not os.path.exists("results_3d/" + args.config):
        os.mkdir("results_3d/" + args.config)

    # save and load path
    LOGFOLDER = args.save_path
    if not os.path.exists(LOGFOLDER):
        os.mkdir(LOGFOLDER)

    if args.useAlist:
        Alist = readAlist(args.Alist_path)
    else:
        Alist = None

    Envs_init(args)
    UCTs_init(args, arealist__=Alist)
    util_init(args)

    bestreward, pbest, ebest = UCTSearch(p, e)

    print("bestreward =", bestreward)
    print(reward_fun(pbest, ebest))

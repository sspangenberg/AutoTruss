import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from truss_envs.reward import Envs_init, reward_fun
from utils.utils import readFile

time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())


def main(args):
    p, e = readFile(os.path.join(args.input_path_2, args.run_id, "_best.txt"))
    Envs_init(args)
    drawGraph(p, e, args)


def drawGraph(p, e, args, canshow=1, reward=0.0, FILENAME="output"):
    print(reward_fun(p, e, mode="check"))
    if reward == 0.0:
        reward, _, _, _, _, _ = reward_fun(p, e)

    if args.env_dims == 3:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        X = np.ones(len(p))
        Y = np.ones(len(p))
        Z = np.ones(len(p))

        def rotate(angle):
            ax.view_init(azim=angle)

        for i in range(len(p)):
            ax.scatter3D([p[i].vec.x], [p[i].vec.y], [p[i].vec.z], color="b")
            X[i] = p[i].vec.x
            Y[i] = p[i].vec.y
            Z[i] = p[i].vec.z

        for i in range(len(e)):
            x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
            y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]
            z0 = [p[e[i].u].vec.z, p[e[i].v].vec.z]
            if e[i].area < 0:
                continue

            if e[i].stress < 0:
                # ax.plot3D(x0, y0, z0, color='g', linewidth = e[i].area / 0.001)
                ax.plot3D(x0, y0, z0, color="g", linewidth=1)
            elif e[i].stress > 0:
                # ax.plot3D(x0, y0, z0, color='r', linewidth = e[i].area / 0.001)
                ax.plot3D(x0, y0, z0, color="r", linewidth=1)
            else:
                ax.plot3D(x0, y0, z0, color="k", linewidth=1)

            # scat = ax.scatter(X, Y, Z)
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array(
                [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
            ).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
                0
            ].flatten() + 0.5 * (X.max() + X.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
                1
            ].flatten() + 0.5 * (Y.max() + Y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
                2
            ].flatten() + 0.5 * (Z.max() + Z.min())

            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], "w")

        # plt.axis("auto")
        # NotImplementedError: Axes3D currently only supports the aspect argument 'auto'. You passed in 'equal'.
        plt.title(str(reward))
        # plot animation gif
        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=100
        )
        FILENAME = os.path.join(
            args.save_path,
            args.run_id,
            args.config
            + "_"
            + str(len(p))
            + "p"
            + str(round(reward, 2))
            + "_3d"
            + ".gif",
        )
        print(FILENAME)
        rot_animation.save(FILENAME, dpi=100)
        if canshow == 1:
            plt.show()

    elif args.env_dims == 2:
        for i in range(len(e)):
            x0 = [p[e[i].u].vec.x, p[e[i].v].vec.x]
            y0 = [p[e[i].u].vec.y, p[e[i].v].vec.y]
            if e[i].stress < 0:
                plt.plot(x0, y0, color="g", linewidth=min(max(e[i].area / 0.005, 2), 6))
            else:
                plt.plot(x0, y0, color="b", linewidth=min(max(e[i].area / 0.005, 2), 6))
        for i in range(len(p)):
            plt.scatter([p[i].vec.x], [p[i].vec.y], color="b", linewidths=5)
        plt.axis("equal")
        plt.title(str(reward))
        FILENAME = os.path.join(
            args.save_path,
            args.run_id,
            args.config
            + "_"
            + str(len(p))
            + "p"
            + str(round(reward, 2))
            + "_2d"
            + ".jpg",
        )
        print(FILENAME)
        plt.savefig(FILENAME, dpi=1000)

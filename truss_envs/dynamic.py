import math
import os
import sys
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as op

from utils.utils import closestDistanceBetweenLines


def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class DynamicModel:
    def __init__(
        self,
        dimension,
        E=193 * 10**9,
        pho=8.0 * 10**3,
        sigma_T=123 * 10**6,
        sigma_C=213 * 10**6,
        dislimit=0.002,
        slenderness_ratio_T=220,
        slenderness_ratio_C=180,
        max_len=5.0,
        min_len=0.03,
        use_self_weight=True,
        use_dis_constraint=True,
        use_stress_constraint=True,
        use_buckle_constraint=True,
        use_slenderness_constraint=True,
        use_longer_constraint=True,
        use_shorter_constraint=True,
        use_cross_constraint=True,
    ):
        self._dimension = dimension
        self._E = E
        self._pho = pho
        self._sigma_T = sigma_T
        self._sigma_C = sigma_C
        self._limit_dis = dislimit
        self.slenderness_ratio_T = slenderness_ratio_T
        self.slenderness_ratio_C = slenderness_ratio_C
        self.max_len = max_len
        self.min_len = min_len
        self._use_self_weight = use_self_weight
        self._use_dis_constraint = use_dis_constraint
        self._use_stress_constraint = use_stress_constraint
        self._use_buckle_constraint = use_buckle_constraint
        self._use_slenderness_constraint = use_slenderness_constraint
        self._use_longer_constraint = use_longer_constraint
        self._use_shorter_constraint = use_shorter_constraint
        self._use_cross_constraint = use_cross_constraint
        print(self._use_self_weight)
        print(self._use_buckle_constraint)

    def _is_struct(self, points, edges):
        total_support = 0
        for p in points:
            if self._dimension == 2:
                total_support += p.supportX + p.supportY
            else:
                total_support += p.supportX + p.supportY + p.supportZ

        if len(points) * self._dimension - len(edges) - total_support > 0:
            return False

        blockPrint()

        op.wipe()
        op.model("basic", "-ndm", self._dimension, "-ndf", self._dimension)

        for i, point in enumerate(points):
            if self._dimension == 2:
                op.node(
                    i,
                    point.vec.x,
                    point.vec.y,
                )
            else:
                op.node(
                    i,
                    point.vec.x,
                    point.vec.y,
                    point.vec.z,
                )

        for i, point in enumerate(points):
            if self._dimension == 2:
                op.fix(
                    i,
                    point.supportX,
                    point.supportY,
                )
            else:
                op.fix(
                    i,
                    point.supportX,
                    point.supportY,
                    point.supportZ,
                )

        op.timeSeries("Linear", 1)
        op.pattern("Plain", 1, 1)

        for i, point in enumerate(points):
            if point.isLoad:
                if self._dimension == 2:
                    op.load(
                        i,
                        point.loadX,
                        point.loadY,
                    )
                else:
                    op.load(
                        i,
                        point.loadX,
                        point.loadY,
                        point.loadZ,
                    )

        op.uniaxialMaterial("Elastic", 1, self._E)

        for i, edge in enumerate(edges):
            op.element("Truss", i, edge.u, edge.v, edge.area, 1)

        if self._use_self_weight:
            gravity = 9.8
            load_gravity = [0 for _ in range(len(points))]

            for i, edge in enumerate(edges):
                edge_mass = edge.len * edge.area * self._pho
                load_gravity[edge.u] += edge_mass * gravity * 0.5
                load_gravity[edge.v] += edge_mass * gravity * 0.5

            for i in range(len(points)):
                if self._dimension == 2:
                    op.load(i, 0.0, -1 * load_gravity[i])
                else:
                    op.load(i, 0.0, 0.0, -1 * load_gravity[i])

        op.system("BandSPD")
        op.numberer("RCM")
        op.constraints("Plain")
        op.integrator("LoadControl", 1.0)
        op.algorithm("Newton")
        op.analysis("Static")
        ok = op.analyze(1)
        if ok < 0:
            ok = False
        else:
            ok = True
        enablePrint()
        return ok

    def _get_dis_value(self, points, mode="check"):
        displacement_weight = np.zeros((len(points), 1))
        for i in range(len(points)):
            if self._dimension == 2:
                weight = max(
                    abs(op.nodeDisp(i, 1)),
                    abs(op.nodeDisp(i, 2)),
                )
            else:
                weight = max(
                    abs(op.nodeDisp(i, 1)),
                    abs(op.nodeDisp(i, 2)),
                    abs(op.nodeDisp(i, 3)),
                )
            displacement_weight[i] = max(weight / self._limit_dis - 1, 0)
        return displacement_weight

    def _get_stress_value(self, edges, mode="check"):
        stress_weight = np.zeros(len(edges))

        for tag, i in enumerate(range(len(edges))):
            edges[i].force = op.basicForce(tag)
            edges[i].stress = edges[i].force[0] / edges[i].area
            if edges[i].stress < 0:
                stress_weight[tag] = max(
                    abs(edges[i].stress) / self._sigma_C - 1.0, 0.0
                )
            else:
                stress_weight[tag] = max(
                    abs(edges[i].stress) / self._sigma_T - 1.0, 0.0
                )
        return stress_weight

    def _get_buckle_value(self, edges):
        buckle_weight = np.zeros(len(edges))
        miu_buckle = 1.0

        for i in range(len(edges)):
            edges[i].force = op.basicForce(i)
            edges[i].stress = edges[i].force[0] / edges[i].area
            if edges[i].stress < 0:
                force_cr = (math.pi**2 * self._E * edges[i].inertia) / (
                    miu_buckle * edges[i].len
                ) ** 2

                buckle_stress_max = force_cr / edges[i].area

                buckle_weight[i] = max(
                    abs(edges[i].stress) / abs(buckle_stress_max) - 1.0, 0.0
                )
        return buckle_weight

    def _get_slenderness_ratio(self, edges):
        lambda_weight = np.zeros(len(edges))

        for i in range(len(edges)):
            edges[i].force = op.basicForce(i)
            edges[i].stress = edges[i].force[0] / edges[i].area
            lambda_weight[i] = max(
                abs(edges[i].len / (edges[i].inertia / edges[i].area) ** 0.5)
                / abs(
                    self.slenderness_ratio_C
                    if edges[i].stress < 0
                    else self.slenderness_ratio_T
                )
                - 1.0,
                0.0,
            )
        return lambda_weight

    def _get_length_longer(self, edges):
        longer_weight = np.zeros(len(edges))

        for i in range(len(edges)):
            longer_weight[i] = max(abs(edges[i].len) / abs(self.max_len) - 1.0, 0.0)
        return longer_weight

    def _get_length_shorter(self, edges):
        shorter_weight = np.zeros(len(edges))

        for i in range(len(edges)):
            if edges[i].len < self.min_len:
                shorter_weight[i] = 1.0 - edges[i].len / self.min_len

        return shorter_weight

    def _get_cross_value(self, points, edges):
        cross_value = 0
        for i in range(len(edges)):
            for j in range(len(edges)):
                if (
                    edges[i].u == edges[j].v
                    or edges[i].u == edges[j].u
                    or edges[i].v == edges[j].v
                    or edges[i].v == edges[j].u
                ):
                    continue
                _, _, dis = closestDistanceBetweenLines(
                    points[edges[i].u].Point2np(),
                    points[edges[i].v].Point2np(),
                    points[edges[j].u].Point2np(),
                    points[edges[j].v].Point2np(),
                    clampAll=True,
                )
                if edges[i].d != None:
                    r1, r2 = edges[i].d / 2, edges[j].d / 2
                else:
                    if self._dimension == 2:
                        r1, r2 = edges[i].area / 2, edges[j].area / 2
                    elif self._dimension == 3:
                        r1, r2 = (
                            np.sqrt(edges[i].area / np.pi),
                            np.sqrt(edges[j].area / np.pi),
                        )
                if dis <= r1 + r2:
                    cross_value += 1
        return cross_value

    def run(self, points, edges, mode="check"):
        if "IS_RUNNING_DYNAMIC" not in os.environ:
            os.environ["IS_RUNNING_DYNAMIC"] = "no"
        while os.environ["IS_RUNNING_DYNAMIC"] == "yes":
            print("waiting for dynamics to be enabled")
            time.sleep(0.1)
        os.environ["IS_RUNNING_DYNAMIC"] = "yes"
        if type(points) == dict or type(points) == OrderedDict:
            _points = []
            for i, point in points.items():
                _points.append(point)
            points = _points

        if type(edges) == dict or type(edges) == OrderedDict:
            _edges = []
            for i, edge in edges.items():
                _edges.append(edge)
            edges = _edges

        is_struct = self._is_struct(points, edges)
        (
            mass,
            dis_value,
            stress_value,
            buckle_value,
            slenderness_value,
            longer_value,
            shorter_value,
            cross_value,
        ) = 0, 0, 0, 0, 0, 0, 0, 0
        for edge in edges:
            mass += edge.len * edge.area * self._pho
        if is_struct:
            if self._use_dis_constraint:
                dis_value = self._get_dis_value(points, mode)
            if self._use_stress_constraint:
                stress_value = self._get_stress_value(edges, mode)
            if self._use_buckle_constraint:
                buckle_value = self._get_buckle_value(edges)
            if self._use_slenderness_constraint:
                slenderness_value = self._get_slenderness_ratio(edges)
            if self._use_longer_constraint:
                longer_value = self._get_length_longer(edges)
            if self._use_shorter_constraint:
                shorter_value = self._get_length_shorter(edges)
        if self._use_cross_constraint:
            cross_value = self._get_cross_value(points, edges)

        if mode != "check":
            if is_struct and self._use_dis_constraint:
                dis_value = dis_value.sum()
            if is_struct and self._use_buckle_constraint:
                buckle_value = buckle_value.sum()
            if is_struct and self._use_slenderness_constraint:
                slenderness_value = slenderness_value.sum()
            if is_struct and self._use_stress_constraint:
                stress_value = stress_value.sum()
            if is_struct and self._use_longer_constraint:
                longer_value = longer_value.sum()
            if is_struct and self._use_shorter_constraint:
                shorter_value = shorter_value.sum()
        os.environ["IS_RUNNING_DYNAMIC"] = "no"
        return (
            is_struct,
            mass,
            dis_value,
            stress_value,
            buckle_value,
            slenderness_value,
            longer_value,
            shorter_value,
            cross_value,
        )

    def render(self, points, edges):
        _ax = plt.axes(projection="3d")
        for point in points.values():
            if point.isSupport:
                _ax.scatter([point.vec.x], [point.vec.y], [point.vec.z], color="g")
            elif point.isLoad:
                _ax.scatter([point.vec.x], [point.vec.y], [point.vec.z], color="r")
            else:
                _ax.scatter([point.vec.x], [point.vec.y], [point.vec.z], color="b")

        for edge in edges.values():
            x0 = [points[edge.u].vec.x, points[edge.v].vec.x]
            y0 = [points[edge.u].vec.y, points[edge.v].vec.y]
            z0 = [points[edge.u].vec.z, points[edge.v].vec.z]

            if edge.stress < -1e-7:
                _ax.plot(
                    x0, y0, z0, color="g", linewidth=(edge.area / math.pi) ** 0.5 * 500
                )
            elif edge.stress > 1e-7:
                _ax.plot(
                    x0, y0, z0, color="r", linewidth=(edge.area / math.pi) ** 0.5 * 500
                )
            else:
                _ax.plot(
                    x0, y0, z0, color="k", linewidth=(edge.area / math.pi) ** 0.5 * 500
                )
        plt.show()

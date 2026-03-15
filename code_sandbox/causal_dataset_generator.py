import numpy as np
import pandas as pd
from collections import deque


class CausalDatasetGenerator:

    def __init__(self, adj):
        self.adj = np.array(adj)
        self.n = len(adj)

    # ---------------------------------------------------
    # traversal order
    # ---------------------------------------------------
    def predecessors_traversal_order(self):

        n = self.n
        adj = self.adj

        parents = [[] for _ in range(n)]
        children = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if adj[i][j] == 1:
                    parents[j].append(i)
                    children[i].append(j)

        indegree = [len(parents[i]) for i in range(n)]

        q = deque(i for i in range(n) if indegree[i] == 0)
        topo = []

        while q:
            node = q.popleft()
            topo.append(node)

            for c in children[node]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    q.append(c)

        preds = [[] for _ in range(n)]
        seen = [set() for _ in range(n)]

        for node in topo:
            for p in parents[node]:

                if p not in seen[node]:
                    seen[node].add(p)
                    preds[node].append(p)

                for anc in preds[p]:
                    if anc not in seen[node]:
                        seen[node].add(anc)
                        preds[node].append(anc)

        return preds, topo

    # ---------------------------------------------------
    # structural equations (symbolic)
    # ---------------------------------------------------
    def build_structural_equations(self, nonlinear=False):

        equations = {}

        for j in range(self.n):

            parents = [i for i in range(self.n) if self.adj[i][j] == 1]

            if not parents:
                equations[j] = f"X{j} = ε{j}"

            else:
                if nonlinear:
                    terms = [f"tanh(w{i}{j}*X{i})" for i in parents]
                else:
                    terms = [f"w{i}{j}*X{i}" for i in parents]

                eq = " + ".join(terms)
                equations[j] = f"X{j} = {eq} + ε{j}"

        return equations

    # ---------------------------------------------------
    # nonlinear causal function
    # ---------------------------------------------------
    def nonlinear_mechanism(self, x, func):

        if func == "tanh":
            return np.tanh(x)

        elif func == "sin":
            return np.sin(x)

        elif func == "sigmoid":
            return 1 / (1 + np.exp(-x))

        elif func == "square":
            return x**2

        else:
            return x

    # ---------------------------------------------------
    # dataset generator
    # ---------------------------------------------------
    def generate_dataset_from_dag(self, n_samples=1000, nonlinear=True):

        preds, topo_order = self.predecessors_traversal_order()
        equations = self.build_structural_equations(nonlinear)

        data = np.zeros((n_samples, self.n))

        weights = np.random.uniform(0.5, 2.0, (self.n, self.n))

        nonlinear_functions = ["tanh", "sin", "sigmoid", "square"]

        for node in topo_order:

            parents = [i for i in range(self.n) if self.adj[i][node] == 1]
            noise = np.random.normal(0, 1, n_samples)

            if not parents:
                data[:, node] = noise

            else:
                value = np.zeros(n_samples)

                for p in parents:

                    effect = weights[p, node] * data[:, p]

                    if nonlinear:
                        func = np.random.choice(nonlinear_functions)
                        effect = self.nonlinear_mechanism(effect, func)

                    value += effect

                data[:, node] = value + noise

        df = pd.DataFrame(data, columns=[f"X{i}" for i in range(self.n)])

        return df, equations
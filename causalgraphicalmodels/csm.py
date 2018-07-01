import inspect
import numpy as np
import pandas as pd
import networkx as nx

from causalgraphicalmodels.cgm import CausalGraphicalModel


class StructuralCausalModel:
    def __init__(self, assignment):
        """
        Creates StructuralCausalModel from assignment of the form
        { variable: Function(parents) }
        """

        self.assignment = assignment.copy()
        nodes = list(assignment.keys())
        set_nodes = []
        edges = []

        for node, model in assignment.items():
            if model is None:
                set_nodes.append(node)

            elif isinstance(model, CausalAssignmentModel):
                edges.extend([
                    (parent, node)
                    for parent in model.parents
                ])

            elif callable(model):
                sig = inspect.signature(model)
                parents = [
                    parent
                    for parent in sig.parameters.keys()
                    if parent != "n_samples"
                ]
                self.assignment[node] = CausalAssignmentModel(model, parents)
                edges.extend([(p, node) for p in parents])

            else:
                raise ValueError("Model must be either callable or None. "
                                 "Instead got {} for node {}."
                                 .format(model, node))

        self.cgm = CausalGraphicalModel(nodes, edges, set_nodes)

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.cgm.dag.nodes())))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))

    def sample(self, n_samples=100, set_values=None):
        """
        Sample from CSM

        Arguments
        ---------
        n_samples: int
            the number of samples to return

        set_values: dict[variable:str, set_value:np.array]
            the values of the interventional variable

        Returns
        -------
        samples: pd.DataFrame
        """
        samples = {}

        if set_values is None:
            set_values = dict()

        for node in nx.topological_sort(self.cgm.dag):
            c_model = self.assignment[node]

            if c_model is None:
                assert len(set_values[node]) == n_samples
                samples[node] = set_values[node]
            else:
                parent_samples = {
                    parent: samples[parent]
                    for parent in c_model.parents
                }
                parent_samples["n_samples"] = n_samples
                samples[node] = c_model(**parent_samples)

        return pd.DataFrame(samples)

    def do(self, node):
        """
        Returns a StructualCausalModel after an intervention on node
        """
        new_assignment = self.assignment.copy()
        new_assignment[node] = None
        return StructuralCausalModel(new_assignment)


class CausalAssignmentModel:
    """
    Basically just a hack to allow me to provide information about the
    arguments of a dynamically generated function.
    """
    def __init__(self, model, parents):
        self.model = model
        self.parents = parents

    def __call__(self, *args, **kwargs):
        assert len(args) == 0
        return self.model(**kwargs)

    def __repr__(self):
        return "CausalAssignmentModel({})".format(",".join(self.parents))


# Some Helper functions for defining models

def _sigma(x):
    return 1 / (1 + np.exp(-x))


def linear_model(parents, weights, offset=0, noise_scale=1):
    """
    Create CausalAssignmentModel for node y of the form
    \sum_{i} x_{i}w_{i} + a + \epsilon

    Arguments
    ---------
    parents: list
        variable names of parents

    weights: list
        weigths of each variable in the sum

    offset: float
        offset for sum

    noise_scale: float
        scale of the normal noise

    Returns
    -------
        model: CausalAssignmentModel
    """
    assert len(parents) == len(weights)
    assert len(parents) > 0
    def model(**kwargs):
        n_samples = kwargs["n_samples"]
        a = np.array([kwargs[p] * w for p, w in zip(parents, weights)], dtype=np.float)
        a = np.sum(a, axis=0)
        a += np.random.normal(loc=offset, scale=noise_scale, size=n_samples)
        return a

    return CausalAssignmentModel(model, parents)


def logistic_model(parents, weights, offset=0):
    """
    Create CausalAssignmentModel for node y of the form
    z = \sum_{i} x_{i}w_{i} + a
    y ~ Binomial(\simga(z))

    Arguments
    ---------
    parents: list
        variable names of parents

    weights: list
        weigths of each variable in the sum

    offset: float
        offset for sum

    Returns
    -------
        model: CausalAssignmentModel
    """
    assert len(parents) == len(weights)
    assert len(parents) > 0
    def model(**kwargs):
        a = np.array([kwargs[p] * w for p, w in zip(parents, weights)])
        a = np.sum(a, axis=0) + offset
        a = _sigma(a)
        a = np.random.binomial(n=1, p=a)
        return a
        

    return CausalAssignmentModel(model, parents)

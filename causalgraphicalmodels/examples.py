"""
A set of example causal systems
"""

import numpy as np

from causalgraphicalmodels.cgm import CausalGraphicalModel
from causalgraphicalmodels.csm import StructuralCausalModel, \
    linear_model, logistic_model


chain = CausalGraphicalModel(
    ["x1", "x2", "x3"],
    [("x1", "x2"), ("x2", "x3")]
)

collider = CausalGraphicalModel(
    ["x1", "x2", "x3"],
    [("x1", "x2"), ("x3", "x2")]
)

fork = CausalGraphicalModel(
    ["x1", "x2", "x3"],
    [("x2", "x1"), ("x2", "x3")]
)

example_path_one = CausalGraphicalModel(
    ["x1", "x2", "x3", "x4", "x5"],
    [("x1", "x2"), ("x2", "x3"), ("x3", "x4"), ("x4", "x5")]
)

sprinkler = CausalGraphicalModel(
    ["season", "rain", "sprinkler", "wet", "slippery"],
    [("season", "rain"), ("season", "sprinkler"), ("rain", "wet"),
     ("sprinkler", "wet"), ("wet", "slippery")]
)

simple_confounded = CausalGraphicalModel(
    ["x", "y", "z"],
    [("z", "x"), ("z", "y"), ("x", "y")]
)

simple_confounded_potential_outcomes = CausalGraphicalModel(
    ["x", "y_0", "y_1", "y", "z"],
    [("z", "x"), ("z", "y_0"), ("z", "y_1"),
     ("y_0", "y"), ("y_1", "y"), ("x", "y")]
)


chain_csm = StructuralCausalModel({
    "a": lambda n_samples: np.ones(n_samples),
    "b": lambda a, n_samples: a + 2,
    "c": lambda b, n_samples: b * 2
})

big_csm = StructuralCausalModel({
    "a": lambda n_samples: np.random.normal(size=n_samples),
    "b": lambda n_samples: np.random.normal(size=n_samples),
    "x": logistic_model(("a", "b"), [-1, 1]),
    "c": linear_model(("x",), [1]),
    "y": linear_model(("c", "e"), [3,-1]),
    "d": linear_model(("b",), [-1]),
    "e": linear_model(("d",), [2]),
    "f": linear_model(("y"), [0.7]),
    "h": linear_model(("y","a"), [1.3, 2.1])
})


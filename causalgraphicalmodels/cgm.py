import networkx as nx
import graphviz
from itertools import combinations, chain


class CausalGraphicalModel:
    """
    Causal Graphical Models
    """

    def __init__(self, nodes, edges, set_nodes=None):
        """
        Create CausalGraphicalModel

        Arguments
        ---------
        nodes: list[node:str]

        edges: list[tuple[node:str, node:str]]

        set_nodes: list[node:str]
        """
        if set_nodes is None:
            self.set_nodes = frozenset()
        else:
            self.set_nodes = frozenset(set_nodes)

        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(nodes)
        self.dag.add_edges_from(edges)

        assert nx.is_directed_acyclic_graph(self.dag)

        for set_node in self.set_nodes:
            # set nodes cannot have parents
            assert not nx.ancestors(self.dag, set_node)

        self.graph = self.dag.to_undirected()

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.dag.nodes())))
        return ("{classname}({vars})"
                .format(classname=self.__class__.__name__,
                        vars=variables))

    def draw(self):
        """
        dot file representation of the CGM.
        """
        dot = graphviz.Digraph()

        for node in self.dag.nodes():
            if node in self.set_nodes:
                dot.node(node, node, {"shape": "ellipse", "peripheries": "2"})
            else:
                dot.node(node, node, {"shape": "ellipse"})

        for a, b in self.dag.edges():
            dot.edge(a, b)

        return dot

    def get_distribution(self):
        """
        Returns a string representing the factorized distribution implied by
        the CGM.
        """
        products = []
        for node in nx.topological_sort(self.dag):
            if node in self.set_nodes:
                continue

            parents = list(self.dag.predecessors(node))
            if not parents:
                p = "P({})".format(node)
            else:
                parents = [
                    "do({})".format(n) if n in self.set_nodes else str(n)
                    for n in parents
                    ]
                p = "P({}|{})".format(node, ",".join(parents))
            products.append(p)
        return "".join(products)

    def do(self, node):
        """
        Apply intervention on node to CGM
        """
        assert node in self.dag.nodes()
        set_nodes = self.set_nodes | frozenset([node])
        nodes = self.dag.nodes()
        edges = [
            (a, b)
            for a, b in self.dag.edges()
            if b != node
            ]
        return CausalGraphicalModel(nodes, edges, set_nodes)

    def _check_d_separation(self, path, zs):
        """
        Check if a path is d-separated by set of variables zs.
        """
        if len(path) < 3:
            return False

        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = self._classify_three_structure(a, b, c)

            if structure in ("chain", "fork") and b in zs:
                return True

            if structure == "collider":
                descendants = (nx.descendants(self.dag, b) | {b})
                if not descendants & set(zs):
                    return True

        return False

    def _classify_three_structure(self, a, b, c):
        """
        Classify three structure as a chain, fork or collider.
        """
        if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
            return "chain"

        if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
            return "chain"

        if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
            return "collider"

        if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
            return "fork"

        raise ValueError("Unsure how to classify ({},{},{})".format(a, b, c))

    def is_d_separated(self, x, y, zs=frozenset()):
        """
        Is x d-separated from y, conditioned on zs?
        """
        assert x in self.dag.nodes()
        assert y in self.dag.nodes()
        assert all([z in self.dag.nodes() for z in zs])

        paths = nx.all_simple_paths(self.graph, x, y)
        return all(self._check_d_separation(path, zs) for path in paths)

    def get_all_independence_relationships(self):
        """
        Returns a list of all pairwise conditional independence relationships
        implied by the graph structure.
        """
        conditional_independences = []
        for x, y in combinations(self.dag.nodes(), 2):
            remaining_variables = set(self.dag.nodes()) - {x, y}
            for cardinality in range(len(remaining_variables) + 1):
                for z in combinations(remaining_variables, cardinality):
                    if self.is_d_separated(x, y, frozenset(z)):
                        conditional_independences.append((x, y, set(z)))

        return conditional_independences

    def get_all_backdoor_paths(self, x, y):
        """
        Get all backdoor paths between x and y
        """
        return [
            path
            for path in nx.all_simple_paths(self.graph, x, y)
            if len(path) > 2
            and path[1] in self.dag.predecessors(x)
            ]

    def is_valid_adjustment_set(self, x, y, z):
        """
        Test whether z is a valid adjustment set for
        estimating the causal impact of x on y via the
        adjustment formula:

        P(y|do(x)) = \sum_{z}P(y|x,z)P(z)

        Arguments
        ---------
        x: str
            Intervention Variable

        y: str
            Target Variable

        z: set
            Adjustment variables

        Returns
        -------
        is_valid_adjustment_set: bool
        """
        assert x in self.dag.nodes()
        assert y in self.dag.nodes()
        assert x not in z
        assert y not in z

        if any([zz in nx.descendants(self.dag, x) for zz in z]):
            return False

        unblocked_backdoor_paths = [
            path
            for path in self.get_all_backdoor_paths(x, y)
            if not self._check_d_separation(path, z)
            ]

        if unblocked_backdoor_paths:
            return False

        return True

    def get_all_backdoor_adjustment_sets(self, x, y):
        """
        Get all sets of variables which are valid adjustment sets for
        estimating the causal impact of x on y via the back door 
        adjustment formula:

        P(y|do(x)) = \sum_{z}P(y|x,z)P(z)

        If there is no such set, returns None.

        Arguments
        ---------
        x: str 
            Intervention Variable 
        y: str
            Target Variable

        Returns
        -------
        condition set: frozenset or None
            Set of variable to condition on or None if no such
            set exists.
        """
        assert x in self.dag.nodes()
        assert y in self.dag.nodes()

        possible_adjustment_variables = (
            set(self.dag.nodes())
            - {x} - {y}
            - set(nx.descendants(self.dag, x))
        )

        valid_adjustment_sets = frozenset([
                                              frozenset(s)
                                              for s in _powerset(
                possible_adjustment_variables)
                                              if
                                              self.is_valid_adjustment_set(x, y,
                                                                           s)
                                              ])

        return valid_adjustment_sets


def _powerset(iterable):
    """
    https://docs.python.org/3/library/itertools.html#recipes
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

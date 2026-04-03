import numpy as np
import pandas as pd
import pytest

from causal_dataset_generator import CausalDatasetGenerator


@pytest.fixture # This fixture runs all the tests with the same simple DAG structure
def simple_dag():
    # 0 → 1, 0 → 2, 1 → 3, 2 → 3
    adj = [
        [0,1,1,0],
        [0,0,0,1],
        [0,0,0,1],
        [0,0,0,0]
    ]
    return adj


def test_predecessors(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    preds, topo = gen.predecessors_traversal_order()

    assert preds[0] == []
    assert preds[1] == [0]
    assert preds[2] == [0]

    # node 3 should include all ancestors
    assert set(preds[3]) == {0,1,2}

    # valid topological order
    assert topo[0] == 0


def test_structural_equations_linear(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    eqs = gen.build_structural_equations(nonlinear=False)

    assert eqs[0] == "X0 = ε0"
    assert "w01*X0" in eqs[1]
    assert "w02*X0" in eqs[2]
    assert "w13*X1" in eqs[3]


def test_structural_equations_nonlinear(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    eqs = gen.build_structural_equations(nonlinear=True)

    assert "tanh" in eqs[1]
    assert "tanh" in eqs[3]


def test_build_linear_model_from_binary_adjacency(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    model = gen.build_linear_model()

    expected = np.array(simple_dag, dtype=float)

    assert np.array_equal(model["coefficient_matrix"], expected)
    assert np.array_equal(model["intercepts"], np.zeros(4))
    assert model["equations"][0] == "X0 = ε0"
    assert model["equations"][3] == "X3 = 1*X1 + 1*X2 + ε3"


def test_build_linear_model_with_custom_weights_and_biases(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    weights = np.array([
        [0.0, 0.5, 1.5, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    biases = np.array([0.0, 1.0, 0.0, -2.0])

    model = gen.build_linear_model(weights=weights, biases=biases)

    assert model["coefficient_matrix"][0, 1] == 0.5
    assert model["coefficient_matrix"][2, 3] == -1.0
    assert model["coefficient_matrix"][0, 3] == 0.0
    assert model["equations"][1] == "X1 = 0.5*X0 + 1 + ε1"
    assert model["equations"][3] == "X3 = 2*X1 + -1*X2 + -2 + ε3"


def test_dataset_shape(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    df, eqs = gen.generate_dataset_from_dag(n_samples=500)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (500, 4)


def test_dataset_columns(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    df, _ = gen.generate_dataset_from_dag(100)

    expected_cols = ["X0", "X1", "X2", "X3"]

    assert list(df.columns) == expected_cols


def test_dataset_no_nan(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    df, _ = gen.generate_dataset_from_dag(200)

    assert not df.isnull().values.any()


def test_nonlinear_generation(simple_dag):

    gen = CausalDatasetGenerator(simple_dag)

    df, _ = gen.generate_dataset_from_dag(200, nonlinear=True)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 4

import pytest
import numpy as np
import os
from distr_nmf.src import tasks_nmf
from distr_nmf.src.exec_config import log_mpc_filename
import luigi
from matrixops.transform import normalize, tfidf
from rri_nmf import nmf


def _gen_random_mat(n, d, density, random_seed=0, nnz_per_row=1):
    np.random.seed(random_seed)
    X = np.zeros((n, d))
    for i in range(n):
        J = np.random.choice(d, size=(1, nnz_per_row), replace=False)
        X[i, J] = 1
        # X[np.indices(I_nz.shape)[0], I_nz] = 1

    density -= float(nnz_per_row) / d
    if density > 0:
        X = X + _mask(np.random.rand(n, d), density)
    return X


def _mask(X, density):
    M = np.random.rand(*X.shape)
    X[M >= density] = 0
    return X





def test_correct_hash(n=100, d=25, seed=0, M=1, n_iter=2):
    """This test may break if new parameters are added to tasks but they are
    not added to the parameter dicts used for testing; this is mostly useful
    to make sure the other tests can still run"""
    X = _gen_random_mat(n, d, 0.1, random_seed=seed)
    idf = True
    tasks_nmf.remove_intermediate = False

    X_fn = '/tmp/X.npy'
    np.save('/tmp/X.npy', X)


    K = 2
    w_row_sum = 1

    nmf_params = {
        "reg_w_l1": 0.0, "project_W_each_iter": True, "random_seed": 0,
        "reg_w_l2": 0.0, "reg_t_l2": 0.0, "k": K, "reg_t_l1": 0.0,
        "project_T_each_iter": True, "agg": "double_precision_nonprivate",
        "init": "random", "t_row_sum": 1.0, "idf": idf,
        "reset_topic_method": "random",
        "w_row_sum": w_row_sum
    }
    dataset_params = {
        "M": M, "d": d, "dataset_name": X_fn, "n": n,
        'execution_mode': 'local'
    }

    Ws = tasks_nmf.GetWeights(dataset_params=dataset_params,
                              nmf_params=nmf_params,
                              n_iter=1,
                              topic_num=K - 1,
                              group_id=0
                              )

    assert Ws.output().path.split('/')[-1] == 'GetWeights__417225404975075603'


# TODO: add fixtures and break down into multiple tests
@pytest.mark.parametrize(('n', 'd', 'seed', 'M', 'n_iter', 'mode'),
                         [
                             (100, 25, 0, 1, 2, 'local'),
                             (100, 25, 0, 1, 2, 'mock_distr_MPC'),
                             (20, 25, 0, 3, 2, 'local'),
                             (21, 26, 1, 5, 2, 'local')
                         ])
def test_distr_matches_centralized(n, d, seed, M, n_iter, mode):
    X = _gen_random_mat(n, d, 0.1, random_seed=seed)
    idf = True
    tasks_nmf.remove_intermediate = False

    X_fn = '/tmp/X.npy'
    np.save('/tmp/X.npy', X)

    for fn in os.listdir('/tmp/'):
        if fn.startswith('Gen') or fn.startswith('Get') or fn.startswith('Agg'):
            try:
                os.remove('/tmp/' + fn)
            except OSError as e:
                raise e
    K = 2
    w_row_sum = 1
    compare_W = True

    if idf:
        X = tfidf(X)
    X = normalize(X)

    LocalNMFTaks = tasks_nmf.MultiWorkerNMF(dataset_name=X_fn,
                                            k=K,
                                            n_iter=n_iter,
                                            idf=idf,
                                            M=M,
                                            execution_mode=mode)
    luigi.build([LocalNMFTaks], local_scheduler=True)

    nmf_params = {
        "reg_w_l1": 0.0, "project_W_each_iter": True, "random_seed": 0,
        "reg_w_l2": 0.0, "reg_t_l2": 0.0, "k": K, "reg_t_l1": 0.0,
        "project_T_each_iter": True, "agg": "double_precision_nonprivate",
        "init": "random", "t_row_sum": 1.0, "idf": idf,
        "reset_topic_method": "random",
        "w_row_sum": w_row_sum
    }
    dataset_params = {
        "M": M, "d": d, "dataset_name": X_fn, "n": n,
        'execution_mode': mode
    }

    for it in range(1, n_iter):
        GT = tasks_nmf.GetTopics(nmf_params=nmf_params,
                                 dataset_params=dataset_params,
                                 n_iter=it, topic_num=K - 1)
        if compare_W:
            if it >= 0:
                W = np.zeros((n, K))
                for m in range(dataset_params['M']):
                    Ws = tasks_nmf.GetWeights(dataset_params=dataset_params,
                                              nmf_params=nmf_params,
                                              n_iter=it,
                                              topic_num=K - 1,
                                              group_id=m
                                              )
                    with Ws.output().open() as f:
                        W_I = np.load(f)
                    I = range(m, n, dataset_params['M'])
                    W[I, :] = W_I

        with GT.output().open() as f:
            T = np.load(f)
        base_nmf_soln = nmf.nmf(X, K, max_iter=it, init='random',
                                random_state=0, debug=0,
                                reset_topic_method='random',
                                fix_reset_seed=True,
                                negative_denom_correction=False,
                                project_W_each_iter=True,
                                w_row_sum=w_row_sum)  # ,
        if compare_W:
            assert np.allclose(W, base_nmf_soln['W'])
        assert np.allclose(T, base_nmf_soln['T'])

def test_MPC_logging():
    #TODO: this depends on being run after test_distr_matches_centralized();
    #TODO: we should use fixtures instead
    with open('/var/log/largenmf_MPC.log') as f:
        s = f.read()
        assert s.count('sending') + s.count('receiving') == 16

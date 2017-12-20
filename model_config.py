import luigi
import copy


def tm_nmf():
    return copy.deepcopy(topic_model_nmf_params)


topic_model_nmf_params = {
    # Distributed computation params
    # -------------------------------------------------------------------------
    'agg': 'double_precision_nonprivate',

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Inference algorithm params
    # -------------------------------------------------------------------------
    'random_seed': 0,
    'init': 'random',  # TODO: implement NNDSVD init
    'reset_topic_method': 'random',  # TODO: better reset

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # NMF Model params
    # -------------------------------------------------------------------------
    'k': 20,  # number of latent components

    'idf': True,  # should only be true for topic modeling

    # W should only be projected for topic modeling, although it helps to get
    #  more orthogonal basis vectors in any application
    'project_W_each_iter': True,
    'w_row_sum': 1.0,
    # T should be projected without loss of generality, although the rows of
    # t should sum to the same thing as the rows of the input X sum to
    'project_T_each_iter': True,
    't_row_sum': 1.0,

    'reg_w_l1': 0.0,
    'reg_w_l2': 0.0,  # negative values may break
    #  the RRI inference algorithm because they can change the sign of the 2nd
    # derivative. Smaller than the smallest l_2 norm of a column of W is safe.
    'reg_t_l1': 0.0,
    'reg_t_l2': 0.0,  # negative values may
    # break. Smaller than the smallest l_2 norm of a row of T is safe.
}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

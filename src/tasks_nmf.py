# imports
# -----------------------------------------------------------------------------
# pip-installable imports
from __future__ import print_function, division
import luigi
import cPickle as pickle
from math import ceil, floor
import numpy as np
import scipy as sp
import os
import logging
import copy
import datetime

# local imports
from model_config import tm_nmf
from exec_config import base_path, log_nmf_filename, log_nmf_level, available_RAM, \
    remove_intermediate
from tasks_MPC import MultiPartyComputationParticipantMixin

# TODO: local dependencies to be included
from matlabinterface import datasets
from matrixops import transform
from experiment_utils.luigi_interface.MTask import \
    AutoLocalOutputMixin, LoadInputDictMixin
from experiment_utils import expm_utils

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


with open(os.path.join(base_path, log_nmf_filename), 'w') as f:
    f.write('New Log\n---')

logging.basicConfig(
    filename=log_nmf_filename,
    level=log_nmf_level, format='%(levelname)s:%(message)s'
)


class MultiWorkerNMF(luigi.WrapperTask):
    dataset_name = luigi.Parameter(default='NIPS', description= \
        'Possible options are : 20NG Reuters [NIPS] Enron Twitter2016 Wiki')


    k = luigi.IntParameter(default=20, description='Number of topics: [20]')
    n_iter = luigi.IntParameter(default=50, description='Number of '
                                                        'iterations: [50]')

    agg = luigi.Parameter(default='double_precision_nonprivate')
    M = luigi.IntParameter(default=-1,
                           description='number of partitions that the input '
                                       'matrix should be split into')

    # TODO: check for config file locally
    execution_mode = luigi.Parameter(default='local', description=\
         'Should we run the NMF locally or in a distributed configuration?'
         '[local] : spawn all workers on the local machine'
         'mock_distr_MPC : pretend to run distributed but use 1 local worker'
         'MLBox_distr : use ~/.MLBox/MLbox.py as configuration and run '
         'distributed using the MLBox package')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Inference algorithm params
    # -------------------------------------------------------------------------
    random_seed = luigi.IntParameter(default=0)
    init = luigi.Parameter(default='random')  # TODO: implement NNDSVD init
    reset_topic_method = luigi.Parameter(default='random')  # TODO: better reset

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # NMF Model params
    # -------------------------------------------------------------------------

    idf = luigi.BoolParameter(default=True)  # should only be true for topic
    # modeling

    # W should only be projected for topic modeling, although it helps to get
    #  more orthogonal basis vectors in any application
    project_W_each_iter = luigi.BoolParameter(default=True)
    w_row_sum = luigi.FloatParameter(default=1.0)
    # T should be projected without loss of generality, although the rows of
    # t should sum to the same thing as the rows of the input X sum to
    project_T_each_iter = luigi.BoolParameter(default=True)
    t_row_sum = luigi.FloatParameter(default=1.0)

    reg_w_l1 = luigi.FloatParameter(default=0.0)
    reg_w_l2 = luigi.FloatParameter(default=0.0)  # negative values may break
    #  the RRI inference algorithm because they can change the sign of the 2nd
    # derivative. Smaller than the smallest l_2 norm of a column of W is safe.
    reg_t_l1 = luigi.FloatParameter(default=0.0)
    reg_t_l2 = luigi.FloatParameter(default=0.0)  # negative values may

    # break. Smaller than the smallest l_2 norm of a row of T is safe.

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __init__(self, *args, **kwargs):
        self.dataset_params = {}

        super(MultiWorkerNMF, self).__init__(*args, **kwargs)

    def requires(self):
        logging.log(logging.INFO, str(self.param_kwargs))
        kwargs_copy = copy.deepcopy(self.param_kwargs)

        assert self.execution_mode in ['local', 'mock_distr_MPC',
                                       'MLBox_distr'],\
            'Unsupported execution mode {}'.format(self.execution_mode)

        dataset_name = kwargs_copy.pop('dataset_name')
        n_iter = kwargs_copy.pop('n_iter')

        nmf_params = {}
        for k in tm_nmf():
            nmf_params[k] = kwargs_copy[k]

        ds = datasets.load_dataset(dataset_name, load_mat=False)
        n, d = ds['size']
        k = self.k
        del ds

        if self.M <= 0:
            mem_WT = (n * k + k * d) * 8
            dpp = int(floor((available_RAM - mem_WT) / (d * 8)))
            M = int(ceil(n / dpp))

        else:
            M = self.M

        self.dataset_params.update( {
            'n': n, 'd': d, 'M': M,
            'dataset_name': dataset_name,
            'execution_mode': self.execution_mode
        })

        logging.log(logging.INFO, 'nmf_params={}\ndataset_params={}'.format(
            nmf_params, self.dataset_params))
        return GetTopics(nmf_params=nmf_params,
                         dataset_params=self.dataset_params,
                         n_iter=n_iter,
                         topic_num=nmf_params['k'] - 1
                         )
        # import pdb; pdb.set_trace()


class RunLocalExperiments(luigi.WrapperTask):
    def requires(self):

        set_dict = expm_utils.generate_settings_dict(
            'distributed_nmf_precision')
        n_iter = set_dict['n_iter'][0]

        for it in range(n_iter):
            reqs = []
            for i in range(expm_utils.n_perms(set_dict)):
                nmf_params = tm_nmf()

                setv = expm_utils.gen_kth_perm(set_dict, i)

                ds = datasets.load_dataset(setv['dataset_name'])
                n, d = ds['X'].shape
                del ds

                nmf_params['random_seed'] = setv.pop('trial_number')
                nmf_params['agg'] = setv['agg']

                docs_per_part = max(setv['k'] * 10, 500)
                M = int(ceil(n / docs_per_part))

                dataset_params = {
                    'n': n, 'd': d, 'M': M,
                    'dataset_name': setv['dataset_name'],
                    'execution_mode': 'local'
                }

                nmf_params['k'] = setv['k']

                reqs.append(EvalFroFitError(nmf_params=nmf_params,
                                            dataset_params=dataset_params,
                                            n_iter=it))

            yield reqs

            # for i in range(0, self.nmf_params['n_iter, 1):
            #     yield GetTopics(n_iter=i, k=NMF_Shared_Params().k,
            # topic_num=19)


def _to_fixed(f, precision=20):
    f = f * (1 << precision)
    return f.astype(np.int64)


def _from_fixed(x, precision=20):
    x = x / (1 << precision)
    return x.astype(np.double)


class EvalFroFitError(AutoLocalOutputMixin(base_path=base_path + '/evals/'),
                      LoadInputDictMixin,
                      luigi.Task
                      ):
    nmf_params = luigi.DictParameter()
    dataset_params = luigi.DictParameter()
    n_iter = luigi.IntParameter()

    def requires(self):
        reqs = {}
        reqs['T'] = GetTopics(nmf_params=self.nmf_params,
                              dataset_params=self.dataset_params,
                              n_iter=self.n_iter,
                              topic_num=self.nmf_params['k'] - 1)
        reqs['W'] = {}
        reqs['X'] = {}
        for m in range(self.dataset_params['M']):
            reqs['W'][m] = GetWeights(dataset_params=self.dataset_params,
                                      nmf_params=self.nmf_params,
                                      n_iter=self.n_iter,
                                      topic_num=self.nmf_params['k'] - 1,
                                      group_id=m
                                      )
            reqs['X'][m] = GenDataset(dataset_params=self.dataset_params,
                                      nmf_params=self.nmf_params,
                                      group_id=m)
        return reqs

    def run(self):
        # TODO: eval for large datasets incrementally (i.e. don't load entire
        #  X at once)
        inp = self.load_input_dict(all_numpy=True)
        n, d, M, k = self.dataset_params['n'], self.dataset_params['d'], \
                     self.dataset_params['M'], self.nmf_params['k']

        W = np.zeros((n, k))
        X = np.zeros((n, d))

        for m in range(M):
            W_I = inp['W'][m]
            X_I = inp['X'][m]
            I = range(m, n, M)
            W[I, :] = W_I
            X[I, :] = X_I

        err = 0.5 * np.sum((X - np.dot(W, inp['T'])) ** 2)
        evals = {'frob_fit_error': err, 'iteration': self.n_iter}
        evals.update(self.nmf_params)
        evals.update(self.dataset_params)

        with self.output().open('w') as f:
            pickle.dump(evals, f, 0)


class GetResidualsFromNetwork(
    AutoLocalOutputMixin(base_path=base_path, output={'wR': 0, 'nw': 0}),
    LoadInputDictMixin,
    MultiPartyComputationParticipantMixin(type='mock_distr_MPC'),
    luigi.Task
):
    nmf_params = luigi.DictParameter()
    dataset_params = luigi.DictParameter()
    n_iter = luigi.IntParameter()
    topic_num = luigi.IntParameter()

    def requires(self):
        if self.dataset_params['M'] > 1:
            raise NotImplementedError(
                'SendResidualsToNetwork doesnt support '
                'more than 1 local worker (M>1) yet.')
        self.k = self.nmf_params['k']
        self.agg = self.nmf_params['agg']
        self.reg_t_l1 = self.nmf_params['reg_t_l1']
        self.reg_t_l2 = self.nmf_params['reg_t_l2']
        self.init = self.nmf_params['init']

        rtv = {}

        if self.n_iter >= 1:

            prev_topic = self.topic_num - 1

            # rolling back index to end of previous loop
            if prev_topic < 0:
                if self.n_iter >= 1:
                    prev_iter = self.n_iter - 1
                prev_topic = self.k - 1
            else:
                prev_iter = self.n_iter

            rtv['resid'] = GetResiduals(
                nmf_params=self.nmf_params,
                dataset_params=self.dataset_params,
                n_iter=prev_iter,
                group_id=0,
                topic_num=prev_topic
            )

        return rtv

    def run(self):
        if self.n_iter >= 1:
            inp = self.load_input_dict(all_numpy=True)

            self.send_to_MPC(inp['resid']['wR'], 'wR')
            self.send_to_MPC(inp['resid']['nw'], 'nw')

            # each receive call will block; potentially these could be done
            # asynchronouslly, but that is more appropriate to be handled within
            # the send/receive implementation
            wR = self.receive_from_MPC('wR')
            nw = self.receive_from_MPC('nw')
        else:
            raise IndexError('Residuals for iteration 0 dont make sense')

        with self.output()['wR'].open('w') as f:
            np.save(f, wR)
        with self.output()['nw'].open('w') as f:
            np.save(f, nw)


class AggregateResiduals(
    AutoLocalOutputMixin(base_path=base_path, output={'numer': 0, 'denom': 0}),
    LoadInputDictMixin,
    luigi.Task
):
    nmf_params = luigi.DictParameter()
    dataset_params = luigi.DictParameter()
    n_iter = luigi.IntParameter()
    topic_num = luigi.IntParameter()

    def requires(self):
        self.k = self.nmf_params['k']
        self.agg = self.nmf_params['agg']
        self.reg_t_l1 = self.nmf_params['reg_t_l1']
        self.reg_t_l2 = self.nmf_params['reg_t_l2']
        self.init = self.nmf_params['init']
        rtv = {}

        if self.n_iter >= 1:

            rtv['resid'] = {}
            prev_topic = self.topic_num - 1

            # rolling back index to end of previous loop
            if prev_topic < 0:
                if self.n_iter >= 1:
                    prev_iter = self.n_iter - 1
                prev_topic = self.k - 1
            else:
                prev_iter = self.n_iter

            # if running locally:
            if self.dataset_params['execution_mode']=='local':
                for m in range(self.dataset_params['M']):
                    rtv['resid'][m] = GetResiduals(
                        nmf_params=self.nmf_params,
                        dataset_params=self.dataset_params,
                        n_iter=prev_iter,
                        group_id=m,
                        topic_num=prev_topic
                    )

                # if running distributed: depend on GetResidualsFromNet which
                # depends on SendResidualsToNet
            elif 'distr' in self.dataset_params['execution_mode']:
                rtv['resid'][0] = GetResidualsFromNetwork(
                                          nmf_params=self.nmf_params,
                                          dataset_params=self.dataset_params,
                                          n_iter=self.n_iter,
                                          topic_num=self.topic_num
                                   )

        yield rtv

    def run(self):
        numer = 0
        denom = 0

        _input = self.load_input_dict(all_numpy=True)

        for m in range(self.dataset_params['M']):  # aggregate the batches
            if self.n_iter >= 1:
                wR = _input['resid'][m]['wR']
                nw = _input['resid'][m]['nw']
            if self.agg == 'double_precision_nonprivate':
                numer += wR
                denom += nw
            elif self.agg.startswith('int_conversion_private'):
                precision = int(self.agg.split('|')[1])
                numer += _to_fixed(wR, precision)
                denom += _to_fixed(nw, precision)

        if self.agg.startswith('int_conversion_private'):
            numer = _from_fixed(numer, precision)
            denom = _from_fixed(denom, precision)

        numer = numer - self.reg_t_l1
        denom = denom + self.reg_t_l2

        with self.output()['numer'].open('w') as f:
            np.save(f, numer)
        with self.output()['denom'].open('w') as f:
            np.save(f, denom)


class GetTopics(AutoLocalOutputMixin(base_path=base_path),
                LoadInputDictMixin,
                luigi.Task
                ):
    nmf_params = luigi.DictParameter()
    dataset_params = luigi.DictParameter()
    n_iter = luigi.IntParameter()
    topic_num = luigi.IntParameter()

    def requires(self):
        self.k = self.nmf_params['k']
        self.agg = self.nmf_params['agg']
        self.reg_t_l1 = self.nmf_params['reg_t_l1']
        self.reg_t_l2 = self.nmf_params['reg_t_l2']
        self.init = self.nmf_params['init']

        if self.n_iter >= 1:
            rtv = {}
            prev_topic = self.topic_num - 1

            # rolling back index to end of previous loop
            if prev_topic < 0:
                if self.n_iter >= 1:
                    prev_iter = self.n_iter - 1
                prev_topic = self.k - 1
            else:
                prev_iter = self.n_iter
            rtv['prev_T'] = GetTopics(nmf_params=self.nmf_params,
                                      dataset_params=self.dataset_params,
                                      n_iter=prev_iter,
                                      topic_num=prev_topic
                                      )

            rtv['Res'] = AggregateResiduals(nmf_params=self.nmf_params,
                                            dataset_params=self.dataset_params,
                                            n_iter=self.n_iter,
                                            topic_num=self.topic_num)

            yield rtv

    def run(self):
        if self.n_iter == 0:
            if self.init == 'random':
                np.random.seed(0)  # random state should be
                # set here since we enter here before GetWeights()
                T = np.random.rand(self.k, self.dataset_params['d'])
                T = transform.normalize(T)
        elif self.n_iter >= 1:
            _input = self.load_input_dict(all_numpy=True)
            T = _input['prev_T']

            numer = _input['Res']['numer']
            denom = _input['Res']['denom']

            T[self.topic_num, :] = np.maximum(numer /
                                              (denom + np.spacing(1)), 0)
            nt1 = np.sum(T[self.topic_num, :])

            if nt1 > 1e-10:
                if self.nmf_params['project_T_each_iter']:
                    T[self.topic_num, :] = transform.euclidean_proj_simplex(
                        T[self.topic_num, :],
                        s=self.nmf_params['t_row_sum']
                    )
            else:
                if self.nmf_params['reset_topic_method'] == 'random':
                    logging.log(logging.INFO, '{0} reseting topic!'.format(
                        self))
                    np.random.seed(self.topic_num)
                    T[self.topic_num, :] = np.random.rand(1, T.shape[1])
                    T[self.topic_num, :] /= T[self.topic_num, :].sum()

        with self.output().open('w') as f:
            np.save(f, T)
            # datasets.dump(T, f, 2)
        if remove_intermediate and self.topic_num == self.k - 1 and \
                        self.n_iter >= 1:
            # remove partial results for this iteration if it's complete
            for t in range(self.k - 1):
                GT = GetTopics(n_iter=self.n_iter,
                               dataset_params=self.dataset_params,
                               nmf_params=self.nmf_params,
                               topic_num=t)
                GT.delete_outputs()

                for m in range(self.dataset_params['M']):
                    GR = GetResiduals(n_iter=self.n_iter,
                                      dataset_params=self.dataset_params,
                                      nmf_params=self.nmf_params,
                                      topic_num=t,
                                      group_id=m
                                      )
                    GR.delete_outputs()

                    GW = GetWeights(n_iter=self.n_iter,
                                    dataset_params=self.dataset_params,
                                    nmf_params=self.nmf_params,
                                    topic_num=t,
                                    group_id=m
                                    )
                    GW.delete_outputs()


class GetResiduals(
    AutoLocalOutputMixin(base_path=base_path, output={'wR': 0, 'nw': 0}),
    LoadInputDictMixin,
    luigi.Task
):
    dataset_params = luigi.DictParameter()
    nmf_params = luigi.DictParameter()
    n_iter = luigi.IntParameter()
    group_id = luigi.IntParameter()
    topic_num = luigi.IntParameter()

    @property
    def resources(self):
        """Put a RAM constraint on ourselves since we will be loading our
        portion of the dataset; prevents out-of-memory error from too many
        jobs trying to load the dataset"""
        # n/M* d * 8 / available_RAM
        if self.dataset_params['execution_mode']=='local':
            n, M, d, k = self.dataset_params['n'], self.dataset_params['M'], \
                         self.dataset_params['d'], self.nmf_params['k']

            dpp = float(n) / M
            mem = dpp * d * 8  # X
            mem += k * d * 8  # T
            mem += dpp * k * 8

            return {'memory': mem / available_RAM}
        return {}

    def requires(self):
        self.k = self.nmf_params['k']

        reqs = {}
        reqs['W'] = GetWeights(dataset_params=self.dataset_params,
                               nmf_params=self.nmf_params,
                               n_iter=self.n_iter,
                               topic_num=self.topic_num,
                               group_id=self.group_id)

        reqs['T'] = GetTopics(dataset_params=self.dataset_params,
                              nmf_params=self.nmf_params,
                              n_iter=self.n_iter,
                              topic_num=self.topic_num)

        reqs['X'] = GenDataset(dataset_params=self.dataset_params,
                               nmf_params=self.nmf_params,
                               group_id=self.group_id)
        return reqs

    def run(self):
        _inp = self.load_input_dict(all_numpy=True)
        logging.log(logging.INFO,
                    '{0}: prev_topic={1} prev_iter={2}'.format(self,
                                                               self.topic_num,
                                                               self.n_iter))
        # residuals using T_{t-1} and W_{t-1} are used for topic t
        next_topic = self.topic_num + 1
        if next_topic >= self.k:
            next_topic = 0
        curr = _get_nmf_residuals(_inp['X'],
                                  _inp['W'],
                                  _inp['T'],
                                  next_topic,
                                  calc_res=True,
                                  calc_W=False)
        with self.output()['wR'].open('w') as f:
            np.save(f, curr['wR'])
        with self.output()['nw'].open('w') as f:
            np.save(f, curr['nw'])


class GetWeights(AutoLocalOutputMixin(base_path=base_path),
                 LoadInputDictMixin,
                 luigi.Task
                 ):
    dataset_params = luigi.DictParameter()
    nmf_params = luigi.DictParameter()
    n_iter = luigi.IntParameter()
    group_id = luigi.IntParameter()
    topic_num = luigi.IntParameter()

    @property
    def resources(self):
        """Put a RAM constraint on ourselves since we will be loading our
        portion of the dataset; prevents out-of-memory error from too many
        jobs trying to load the dataset"""
        # n/M* d * 8 / available_RAM
        if self.dataset_params['execution_mode'] == 'local':
            n, M, d, k = self.dataset_params['n'], self.dataset_params['M'], \
                         self.dataset_params['d'], self.nmf_params['k']

            dpp = float(n) / M
            mem = dpp * d * 8  # X
            mem += k * d * 8  # T
            mem += dpp * k * 8

            return {'memory': mem / available_RAM}
        return {}

    def requires(self):
        self.k = self.nmf_params['k']
        self.init = self.nmf_params['init']
        self.project_W_each_iter = self.nmf_params['project_W_each_iter']
        self.w_row_sum = self.nmf_params['w_row_sum']

        reqs = {}

        if self.n_iter >= 1:
            prev_topic = self.topic_num - 1

            # rolling back index to end of previous loop
            if prev_topic < 0:
                if self.n_iter >= 1:
                    prev_iter = self.n_iter - 1
                prev_topic = self.k - 1
            else:
                prev_iter = self.n_iter

            # most recent W for previous topic
            reqs['W'] = GetWeights(dataset_params=self.dataset_params,
                                   nmf_params=self.nmf_params,
                                   n_iter=prev_iter,
                                   topic_num=prev_topic,
                                   group_id=self.group_id)
            # corresponding most recent T, same topic
            reqs['T'] = GetTopics(dataset_params=self.dataset_params,
                                  nmf_params=self.nmf_params,
                                  n_iter=self.n_iter,
                                  topic_num=self.topic_num)
        reqs['X'] = GenDataset(dataset_params=self.dataset_params,
                               nmf_params=self.nmf_params,
                               group_id=self.group_id)
        yield reqs

    def run(self):
        _input = self.load_input_dict(all_numpy=True)
        X = _input['X']

        # for preparing residuals for computation of T
        next_topic = self.topic_num + 1
        if next_topic >= self.k:
            next_topic = 0

        if 'W' not in _input.keys():
            # done to match random state
            np.random.seed(self.nmf_params['random_seed'])  # random state
            # should be
            # set here since we enter here before GetWeights()
            # doing this just for sake of random state
            T_seed = np.random.rand(self.k, self.dataset_params['d'])
            ds = datasets.load_dataset(self.dataset_params['dataset_name'],
                                       load_mat=False)
            n = ds['size'][0]
            W = np.random.rand(n, self.k)
            W = W[range(self.group_id, n, self.dataset_params['M']), :]
        else:
            T = _input['T']
            W = _input['W']
            curr = _get_nmf_residuals(X, W, T, self.topic_num,
                                      reg_w_l1=self.nmf_params['reg_w_l1'],
                                      reg_w_l2=self.nmf_params['reg_w_l2'],
                                      calc_W=True,
                                      calc_res=False)
            W = curr['W']

            nw1 = np.sum(W[:, self.topic_num])

            if True or nw1 > 1e-10:
                pass
            else:
                if self.nmf_params['reset_topic_method'] == 'random':
                    logging.log(logging.INFO, '{0} reseting weights!'.format(
                        self))
                    np.random.seed(self.topic_num)

                    W[:, self.topic_num] = np.random.rand(W.shape[0])
                    W[:, self.topic_num] /= W[:, self.topic_num].sum()
                    # project the W's at the end of an iteration of updating
                    # all the
                    # topics
        if self.project_W_each_iter and not self.w_row_sum is \
                None and self.topic_num == self.k - 1:
            logging.log(logging.INFO, '{0} projecting rows of W onto '
                                      'simplex'.format(self))
            W = transform.proj_mat_to_simplex(W, s=self.w_row_sum)

        with self.output().open('w') as f:
            np.save(f, W)


class GenDataset(AutoLocalOutputMixin(base_path=base_path),
                 LoadInputDictMixin,
                 luigi.Task
                 ):
    group_id = luigi.IntParameter()
    dataset_params = luigi.DictParameter()
    nmf_params = luigi.DictParameter()

    resources = {'memory': 1}  # use all available memory, so we dont get an

    # out-of-RAM error when just spliting up the dataset


    def requires(self):
        self.idf = self.nmf_params['idf']
        self.row_normalize = self.nmf_params['project_T_each_iter']
        self.n_groups = self.dataset_params['M']

        if self.idf:
            return GetTFIDF(self.dataset_params['dataset_name'])  # global IDF
            # weights to be
            # used on each local part

    def run(self):
        ds = datasets.load_dataset(self.dataset_params['dataset_name'])
        n, d = ds['X'].shape
        X_I = ds['X'][range(self.group_id, n, self.n_groups), :]
        if sp.sparse.issparse(X_I):
            X_I = X_I.toarray()
        # get correctly strided rows

        if self.idf:
            with self.input().open() as f:
                idf = pickle.load(f)
            X_I = X_I * idf
        if self.row_normalize:
            X_I = transform.normalize(X_I)

        with self.output().open('w') as f:
            np.save(f, X_I)


class GetTFIDF(AutoLocalOutputMixin(base_path=base_path),
               LoadInputDictMixin,
               luigi.Task
               ):
    # TODO; currently this is global IDF weights (which is correct for a
    # centralized dataset being computed my multiple workers) but not correct
    #  for a comparing a distributed computation vs centralized, since each
    # party will compute their own IDF; or we can have a pre-step where we
    # compute the global IDF weights by secure sum.
    dataset_name = luigi.Parameter()

    resources = {'memory': 1.0}

    def run(self):
        ds = datasets.load_dataset(self.dataset_name)
        n = ds['X'].shape[0]
        df = (ds['X'] > 0).sum(0)
        idf = np.log(n / (df + np.spacing(1)))
        if sp.sparse.issparse(idf):
            idf = idf.toarray()
        if type(idf) == np.matrix:
            idf = np.asarray(idf)
        with self.output().open('w') as f:
            pickle.dump(idf, f, 2)


def _get_nmf_residuals(X, W, T, t, reg_w_l1=0, reg_w_l2=0, calc_res=False,
                       calc_W=False):
    """
    Compute residuals and topic weights.

    Parameters
    ----------
    X
    W
    T
    t
    reg_w_l1
    reg_w_l2
    calc_res
    calc_W

    Returns
    -------

    """
    rtv = {}

    # get residuals for calculation of T
    if calc_res:
        w = W[:, t]
        wX = w.T.dot(X)
        wW = w.T.dot(W)
        wW[t] = 0
        rtv['wR'] = wX - wW.dot(T)
        rtv['nw'] = (W[:, t] ** 2).sum()  # ||W[:, t]||^2, this is a scalar

    # get W
    if calc_W:
        Xt = X.dot(T[t, :].T)
        Tt = T.dot(T[t, :].T)
        Tt[t] = 0
        Rt = Xt - W.dot(Tt)
        nt = (T[t, :] ** 2).sum()  # ||T[t, :]||^2, a scalar
        numer = Rt - reg_w_l1
        denom = nt + reg_w_l2
        W[:, t] = np.maximum(numer / (denom + np.spacing(1)), 0)

        rtv['W'] = W

    return rtv


if __name__ == '__main__':
    T = MultiWorkerNMF()
    luigi.build([T])

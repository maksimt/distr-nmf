from __future__ import print_function
from abc import ABCMeta, abstractmethod




class MetaMPCMixin(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def send_to_MPC(self, obj, id):
        """Send obj to the MPC computation identified by id"""
        raise NotImplementedError('This class should be subclassed and '
                                  'implemented')

    @abstractmethod
    def receive_from_MPC(self, id):
        """Receive the result of the MPC for objects identified by id"""
        raise NotImplementedError('This class should be subclassed and '
                                  'implemented')

class MockMPCMixin(MetaMPCMixin):

    def send_to_MPC(self, obj, id):
        self.mpcs[id] = obj
        print('mock_MPC-sending {id} of size {size}'.format(id=id, size=obj.size))

    def receive_from_MPC(self, id):
        print('mock_MPC-receiving {id} from MPC'.format(id=id))
        return self.mpcs[id]


def MultiPartyComputationParticipantMixin(type='mock_distr_MPC'):
    if type == 'mock_distr_MPC':
        class T(MockMPCMixin):
            pass
        T.mpcs = {}

        return T
    else:
        raise NotImplementedError(('MPC of type {type} not '+
                                  'implemented.').format(type=type))
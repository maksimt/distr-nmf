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

class DummyMPCMixin(MetaMPCMixin):
    def __init__(self):
        self.mpcs = {}

    def send_to_MPC(self, obj, id):
        self.mpcs[id] = obj
        print('Dummy-sending {id} of size {size}'.format(id=id, size=obj.size))

    def receive_from_MPC(self, id):
        print('Dummy-receiving {id} from MPC'.format(id=id))
        return self.mpcs[id]


def MultiPartyComputationParticipantMixin(type='dummy'):
    if type == 'dummy':
        class T(DummyMPCMixin):
            pass

        return T
    else:
        raise NotImplementedError(('MPC of type {type} not '+
                                  'implemented.').format(type=type))
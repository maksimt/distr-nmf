




class MultiPartyComputationParticipantMixin(object):

    def send_to_MPC(self, obj, id):
        """Send obj to the MPC computation identified by id"""
        raise NotImplementedError('This class should be subclassed and '
                                  'implemented')

    def receive_from_MPC(self, id):
        """Receive the result of the MPC for objects identified by id"""
        raise NotImplementedError('This class should be subclassed and '
                                  'implemented')
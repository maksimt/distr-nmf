import logging

# Global Execution params
# -----------------------------------------------------------------------------
base_path = '/tmp/'  # where should intermediate
# results be stored?

remove_intermediate = True  # remove intermediate results to save space?
# Should be True for actual runs, False for testing

# base_path = '/dev/shm/largenm/'
available_RAM = 4e9  # How much RAM should we use in bytes? 4e9 = 4GB a
# good idea is to under-provision by 30-50% of available system memory

log_nmf_filename = '/var/log/largenmf.log'  # will be in the base_path directory
log_nmf_level = logging.WARNING
log_mpc_filename = '/var/log/largenmf_MPC.log'
log_mpc_level = logging.INFO
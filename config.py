# Global Execution params
# -----------------------------------------------------------------------------
base_path = '/mnt/WD8TB/experiments/largenmf/'  # where should intermediate
# results be stored?

remove_intermediate = True  # remove intermediate results to save space?
# Should be True for actual runs, False for testing

# base_path = '/dev/shm/largenm/'
available_RAM = 64e9  # How much RAM should we use in bytes? 64e9 = 64GB a
# good idea is to under-provision by 30-50% of available system memory

log_filename = 'largenmf.log'  # will be in the base_path directory

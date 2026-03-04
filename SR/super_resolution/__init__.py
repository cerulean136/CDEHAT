import os
import sys
import os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")  # Set the filter to ignore all warnings.
    os.environ["PYTHONWARNINGS"] = "ignore"  # Set environment variables to turn off child process warnings.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid initializing multiple OpenMP runtime libraries with errors.

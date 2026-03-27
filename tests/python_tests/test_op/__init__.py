"""Joy GPU backend single-operator unit tests.

Each test module under this package exercises one operator from
``joy/lib/backend/gpu`` end-to-end:

  * Allocate input tensors on the GPU through the test runtime helpers
  * Build the corresponding ``MemrefDesc`` array and ``GpuContext``
  * Call the ``joy_gpu_<op>`` extern "C" entry point exported by the
    shared library ``libjoy_gpu_runtime.so``
  * Copy results back and compare against a NumPy / PyTorch reference

All tests can be run via ``python3 run_all.py`` in this directory.
"""

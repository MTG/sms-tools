import sys
import types


def _install_utilfunctions_c_stub():
    package_name = "smstools.models.utilFunctions_C"
    module_name = "smstools.models.utilFunctions_C.utilFunctions_C"

    if module_name in sys.modules:
        return

    package_module = sys.modules.get(package_name)
    if package_module is None:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []
        sys.modules[package_name] = package_module

    stub_module = types.ModuleType(module_name)

    def genSpecSines(iploc, ipmag, ipphase, N):
        import numpy as np

        return np.zeros(N, dtype=complex)

    def twm(pfreq, pmag, f0cf):
        import numpy as np

        if f0cf.size == 0:
            return 0.0, np.inf
        return float(f0cf[np.argmax(f0cf > 0)] if np.any(f0cf > 0) else f0cf[0]), 0.0

    stub_module.genSpecSines = genSpecSines
    stub_module.twm = twm
    package_module.utilFunctions_C = stub_module
    sys.modules[module_name] = stub_module


_install_utilfunctions_c_stub()

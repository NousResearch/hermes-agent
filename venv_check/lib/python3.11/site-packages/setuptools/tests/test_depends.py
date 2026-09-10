import sys

from setuptools import depends


class TestGetModuleConstant:
    def test_basic(self):
        """
        Invoke get_module_constant on a module in
        the test package.
        """
        mod_name = 'setuptools.tests.mod_with_constant'
        val = depends.get_module_constant(mod_name, 'value')
        assert val == 'three, sir!'
        assert 'setuptools.tests.mod_with_constant' not in sys.modules

from jaraco.test.cpython import from_test_support, try_import

os_helper = try_import('os_helper') or from_test_support('can_symlink')

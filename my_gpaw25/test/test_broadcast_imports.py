import pytest
from my_gpaw25._broadcast_imports import broadcast_imports

# These tests would be better if we could find a way to call them
# without first executing my_gpaw25.__init__ (e.g. with pytest-forked or
# something?).
#
# However at least the bad() test is useful because it specifies the
# fail behaviour which may not otherwise be exercised.
#
# Some day: Execute the imports in such a way that we can verify the
# module counts.


def test_bcast_imports_good():
    with broadcast_imports:
        import my_gpaw25.poisson as poisson_module
        assert poisson_module is not None


def test_bcast_imports_bad():
    with broadcast_imports:
        with pytest.raises(ModuleNotFoundError):
            import my_gpaw25.module_not_to_be_found  # noqa

"""Check that our tab-completion script has been updated."""
from ase.cli.completion import update
from my_gpaw25.cli.completion import path
from my_gpaw25.cli.main import commands


def test_complete():
    try:
        update(path, commands, test=True)
    except ValueError:
        raise ValueError(
            'Please update gpaw/cli/complete.py using '
            '"python3 -m my_gpaw25.test.test_complete".')


if __name__ == '__main__':
    update(path, commands, test=False)

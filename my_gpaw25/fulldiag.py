from my_gpaw25 import GPAW


def fulldiag(filename, bands=None, scalapack=1):
    """Set up full H and S matrices and find all or some eigenvectors/values.

    calc: str
        Filename of gpw-file.
    bands: int
        Number of bands to calculate.  Defaults to all.
    scalapack: int
        Number of cores to use for ScaLapack.  Default is one.
    dry_run: bool
        Don't do actual calculation.

    """
    name, ext = filename.rsplit('.', 1)
    assert ext == 'gpw'
    calc = GPAW(filename,
                parallel={'band': scalapack},
                txt=name + '-all.txt')
    calc.diagonalize_full_hamiltonian(bands)
    calc.write(name + '-all.gpw', 'all')

    ng = calc.wfs.pd.ngmax
    mem = ng**2 * 16 / 1024**2
    print('Maximum matrix size: {0}x{0} = {1:.3f} MB'.format(ng, mem))


class CLICommand:
    """Calculate all or some eigenvectors/values for fixed H."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')
        parser.add_argument('-b', '--bands', type=int)
        parser.add_argument('-s', '--scalapack', type=int, default=1)

    @staticmethod
    def run(args):
        fulldiag(args.gpw, args.bands, args.scalapack)

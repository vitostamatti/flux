from IPython.core.magics.code import extract_symbols
from flux.utils.utils import _isnotebook, _getfile
import inspect
from flux.core.flux import Flux

   
class export_function:
    """
    Useful decorator to export functions that
    are nested inside other functions.

    Ones Flux is fitted and working on a notebook environment,
    if you need to use it in a new environment, you'll also need
    all imported and used functions. This decorator automate that
    task.

    Args:
        flux (Flux): Flux to which export function
    """
    def __init__(self, flux:Flux):
        if not _isnotebook():
            raise Exception("export_function decorator can only be used in jupyter notebooks")
        self._flux = flux

    def _save_source_code(self, f):
        cell_code = "".join(inspect.linecache.getlines(_getfile(f)))
        sc = extract_symbols(cell_code, f.__name__)[0][0]
        self._flux._register_source_code(f.__name__,sc)

    def __call__(self, f):
        self._save_source_code(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
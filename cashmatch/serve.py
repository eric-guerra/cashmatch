"""Demo entry point: trains on synthetic history at startup. Production loads a versioned model artifact."""
from . import synth
from .api import create_app
from .model import Matcher

_inv, _pay, _truth = synth.generate(seed=1, n_payments=500)
app = create_app(Matcher().fit(_pay, _inv, _truth))

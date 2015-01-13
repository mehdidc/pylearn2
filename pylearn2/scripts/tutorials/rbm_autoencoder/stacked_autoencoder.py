
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.modem import Model
from pylearn2.space import VectorSpace

class StackedAutoencodersModel(Model):

    def __init__(self, nvis=None, *args, **kwargs):
        assert nvis is not None

        super(MyModelSubclass, self).__init__()

        self.nvis = nvis

        self._params = [

        ]
        self.input_space = VectorSpace(dim=self.nvis)

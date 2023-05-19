class MatrixSet:
    def __init__(self, IuA, elevI, gridA, hcdefs):
        self.IuA = IuA
        self.elevI = elevI
        self.gridA = gridA
        self.hcdefs = hcdefs

    @cached_property
    def IuE(self):
        return ecmaker.extend_to_elev(self.IuA, self.elevI, self.hcdefs)

    @cached_property
    def IvA(self):
        return ecmaker.scale(self.IuA)

    @cached_property
    def AvI(self):
        return ecmaker.scale(self.IuA.transpose())

    @cached_property
    def IvE(self):
        return ecmaker.scale(self.IuE)

    @cached_property
    def EvI(self):
        return ecmaker.scale(self.IuE.transpose())

    @cached_property
    def EvA(self):
        return self.EvI * self.IvA

    @cached_property
    def AvE(self):
        return self.AvI * self.IvE

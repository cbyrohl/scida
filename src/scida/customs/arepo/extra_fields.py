# convenience file to add common fields whenever applicable


class FieldDef:
    def __init__(self, fieldname, dependencies, parttype):
        self.fieldname = fieldname
        self.dependencies = dependencies
        self.parttype = parttype

    def __call__(self, func):
        self.func = func

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


class FieldDefs:
    def __init__(self):
        self.fielddefs = {}

    def register_field(self, fieldname, dependencies, parttype):
        self.fielddefs[fieldname] = FieldDef(fieldname, dependencies, parttype)
        return self.fielddefs[fieldname]

    def __getitem__(self, fieldname):
        return self.fielddefs[fieldname]


fielddefs = FieldDefs()


@fielddefs.register_field("Temperature", ["ElectronAbundance", "InternalEnergy"], "PartType0")
def Temperature(arrs, ureg=None, **kwargs):
    """Compute gas temperature given (ElectronAbundance,InternalEnergy) in [K]."""
    xh = 0.76
    gamma = 5.0 / 3.0

    m_p = 1.672622e-24  # proton mass [g]
    k_B = 1.380650e-16  # boltzmann constant [erg/K]

    UnitEnergy_over_UnitMass = 1e10  # standard unit system (TODO: can obtain from snapshot)
    f = UnitEnergy_over_UnitMass
    if ureg is not None:
        f = 1.0
        m_p = m_p * ureg.g
        k_B = k_B * ureg.erg / ureg.K
    else:
        # in this case the arrs cannot have pint units
        assert not hasattr(arrs["ElectronAbundance"], "units")
        assert not hasattr(arrs["InternalEnergy"], "units")

    xe = arrs["ElectronAbundance"]
    u_internal = arrs["InternalEnergy"]

    mu = 4 / (1 + 3 * xh + 4 * xh * xe) * m_p
    temp = f * (gamma - 1.0) * u_internal / k_B * mu

    return temp

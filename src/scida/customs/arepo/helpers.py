def grp_type_str(gtype):
    if str(gtype).lower() in ["group", "groups", "halo", "halos"]:
        return "halo"
    if str(gtype).lower() in ["subgroup", "subgroups", "subhalo", "subhalos"]:
        return "subhalo"
    raise ValueError("Unknown group type: %s" % gtype)


def part_type_num(ptype):
    """Mapping between common names and numeric particle types."""
    ptype = str(ptype).replace("PartType", "")
    if ptype.isdigit():
        return int(ptype)

    if str(ptype).lower() in ["gas", "cells"]:
        return 0
    if str(ptype).lower() in ["dm", "darkmatter"]:
        return 1
    if str(ptype).lower() in ["dmlowres"]:
        return 2  # only zoom simulations, not present in full periodic boxes
    if str(ptype).lower() in ["tracer", "tracers", "tracermc", "trmc"]:
        return 3
    if str(ptype).lower() in ["star", "stars", "stellar"]:
        return 4  # only those with GFM_StellarFormationTime>0
    if str(ptype).lower() in ["wind"]:
        return 4  # only those with GFM_StellarFormationTime<0
    if str(ptype).lower() in ["bh", "bhs", "blackhole", "blackholes", "black"]:
        return 5
    if str(ptype).lower() in ["all"]:
        return -1

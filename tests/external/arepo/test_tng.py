import logging

import dask.array as da
import numpy as np
import pytest

from scida import load
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_groupedoperations(testdatapath):
    snp = load(testdatapath, units=True)
    g = snp.grouped("Masses")
    boundmass = g.sum().evaluate().sum()
    boundmass2 = da.sum(
        snp.data["PartType0"]["Masses"][: np.sum(snp.get_grouplengths())]
    ).compute()
    assert boundmass.units == boundmass2.units
    assert np.isclose(boundmass, boundmass2)
    assert np.sum(g.half().sum().evaluate()) < np.sum(g.sum().evaluate())
    assert np.allclose(
        g.apply(lambda x: x[::2]).sum().evaluate(), g.half().sum().evaluate()
    )
    g2 = snp.grouped()

    def customfunc1(arr, fieldnames="Masses"):
        return arr[::2]

    s = g2.apply(customfunc1).sum()
    assert np.allclose(s.evaluate(), g.half().sum().evaluate())
    arr = snp.data["PartType0"]["Density"] * snp.data["PartType0"]["Masses"]
    boundvol2 = snp.grouped(arr).sum().evaluate().sum()
    units = arr.units
    assert 0.0 * units < boundvol2 < 1.0 * units

    def customfunc2(dens, mass, fieldnames=["Density", "Masses"]):
        return dens * mass

    s = g2.apply(customfunc2).sum()
    boundvol = s.evaluate().sum()
    assert np.isclose(boundvol, boundvol2)
    m = snp.grouped("Masses").sum().evaluate()
    m10 = snp.grouped("Masses").sum().evaluate(nmax=10)
    assert m10.shape[0] == 10
    assert np.allclose(m[:10], m10)
    idxlist = [3, 5, 7, 25200]
    m4 = snp.grouped("Masses").sum().evaluate(idxlist=idxlist)
    assert m4.shape[0] == len(idxlist)
    assert np.allclose(m[idxlist], m4)
    nsubs = snp.data["Subhalo"]["SubhaloMass"].shape[0]
    sm = snp.grouped("Masses", objtype="subhalos").sum().evaluate()
    assert sm.shape[0] == nsubs
    h_bhmass = (
        snp.grouped("BH_Mass", objtype="halos", parttype="PartType5").sum().evaluate()
    )
    h_bhmass_true = snp.data["Group"]["GroupBHMass"].compute()
    v1, v2 = h_bhmass.magnitude, h_bhmass_true.magnitude
    assert np.allclose(v1, v2)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_groupedoperations_subhalo(testdatapath):
    snp = load(testdatapath)
    sh_bhmass = (
        snp.grouped("BH_Mass", objtype="subhalos", parttype="PartType5")
        .sum()
        .evaluate()
    )
    sh_bhmass_true = snp.data["Subhalo"]["SubhaloBHMass"].compute()
    v1, v2 = sh_bhmass.magnitude, sh_bhmass_true.magnitude
    assert np.allclose(v1, v2)
    dmparticle_mass = snp.header["MassTable"][1] * snp.ureg("code_mass")
    ones = np.ones(snp.data["PartType1"]["Coordinates"].shape[0]) * dmparticle_mass
    sh_dmmass = (
        snp.grouped(ones, objtype="subhalos", parttype="PartType1").sum().evaluate()
    )
    sh_dmmass_true = (
        snp.data["Subhalo"]["SubhaloLenType"][:, 1].compute() * dmparticle_mass
    )
    v1, v2 = sh_dmmass.magnitude, sh_dmmass_true.magnitude
    assert np.allclose(v1, v2)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_interface_groupedoperations_nonscalar(testdatapath, caplog):
    """Test grouped operations with non-scalar function outputs."""
    snp = load(testdatapath)
    ngrp = snp.data["Group"]["GroupMass"].shape[0]
    g = snp.grouped()
    shape = (2,)

    def customfunc(mass, fieldnames=["Masses"], shape=shape):
        return np.array([np.min(mass), np.max(mass)])

    s = g.apply(customfunc)
    res = s.evaluate()
    assert res.shape[0] == ngrp
    assert res.shape[1] == shape[0]
    assert np.all(res[:, 0] <= res[:, 1])

    def customfunc2(mass, fieldnames=["Masses"]):
        return np.array([np.min(mass), np.max(mass)])

    s = g.apply(customfunc2)
    res = s.evaluate()
    assert res.shape[1] == shape[0]

    def customfunc3(mass, fieldnames=["Masses"]):
        return [mass[2], mass[3]]

    s = g.apply(customfunc3)
    caplog.set_level(logging.WARNING)
    try:
        res = s.evaluate()
    except IndexError:
        pass
    assert "Exception during shape inference" in caplog.text


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_default_recipes(testdatapath):
    obj = load(testdatapath)
    T = obj.data["PartType0"]["Temperature"][0].compute()
    v1 = T.to_base_units().magnitude
    obj2 = load(testdatapath, units=False)
    v2 = obj2.data["PartType0"]["Temperature"][0].compute()
    assert np.allclose(v1, v2)


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_allunitsdiscovered(testdatapath, caplog):
    load(testdatapath)
    caplog.set_level(logging.DEBUG)
    assert (
        "Cannot determine units from neither unit file nor metadata" not in caplog.text
    )

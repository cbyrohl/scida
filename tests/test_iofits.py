def test_fitsread():
    path = "/fastdata/public/testdata-scida/specObj-dr16.fits"

    from astropy.io import fits

    ext = 1
    with fits.open(path, memmap=True, mode="denywrite") as hdulist:
        arr = hdulist[ext].data
    print(arr.dtype)
    print(arr.memmap)
    print(arr[:10]["SURVEY"])
    # print(arr["SURVEY"][:10])
    assert arr is not None
    # test = read(path, ext=1)

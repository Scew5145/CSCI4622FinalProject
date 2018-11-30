import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# Convert an entry's data into a polynomial using polyfit
# Output values are the "new features". These features represent a sort of best guess for what type of function would
# produce the output data. May need to tweak the polynomial count,
# may be good to tweak it for each spectrum individually


def poly_extract(mjd, flux):
    """
    :param ids: array of object ids
    :param mjd: 2d array timestamps (aka x)
    :param flux: 2d array flux entries (aka y)
    :return: array of polynomial coefficients
    """
    # TODO: Flux error. Use it?
    coefs = []
    assert len(mjd) == len(flux)

    for i in range(len(mjd)):
        earliest_time = min(mjd[i])
        assert len(mjd[i]) == len(flux[i])
        coefs.append(np.polyfit(np.subtract(mjd[i], earliest_time), flux[i], 3))

    return coefs


ex_set = pd.read_csv('training_set.csv')
mjd = []
flux = []
ids = []
i = 0
while True:
    current_id = ex_set["object_id"][i]
    # print(current_id)
    temp_mjd = []
    temp_flux = []
    # if i == len(ex_set['object_id']):
    if current_id == 16496:
        break
    while ex_set["object_id"][i] == current_id:
        temp_flux.append(ex_set["flux"][i])
        temp_mjd.append(ex_set["mjd"][i])
        i += 1
    ids.append(current_id)
    mjd.append(temp_mjd)
    flux.append(temp_flux)

# print(ids[0], flux[0], mjd[0])
print(poly_extract(mjd, flux))


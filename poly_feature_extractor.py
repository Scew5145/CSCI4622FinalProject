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
    :param mjd: 3d array timestamps (aka x)
    :param flux: 3d array flux entries (aka y)
    :return: array of polynomial coefficients
    """
    # TODO: Flux error. Use it?
    coefs = []
    assert len(mjd) == len(flux)

    for i in range(len(mjd)):
        if i%1000 == 0:
            print("object", i)
        temp_pb_dict = {}
        for pb in mjd[i]:
            pb_coefs = []

            assert len(mjd[i][pb]) == len(flux[i][pb])
            earliest_time = min(mjd[i][pb])
            pb_coefs = np.polyfit(np.subtract(mjd[i][pb], earliest_time), flux[i][pb], 3)
            temp_pb_dict[pb] = pb_coefs
        coefs.append(temp_pb_dict)

    return coefs


ex_set = pd.read_csv('training_set.csv')
mjd = []
flux = []
ids = []
i = 0
set_length = len(ex_set['object_id'])
done = False
while not done:
    current_id = ex_set["object_id"][i]
    # print(current_id)
    temp_mjd = {}
    temp_flux = {}
    # if current_id > 16339:
        # Can stop this early here
        # print(i, "lines parsed. Starting polyfit")
        # break
    while ex_set["object_id"][i] == current_id:
        passband = ex_set["passband"][i]
        if passband not in temp_flux:
            temp_flux[passband] = [ex_set["flux"][i]]
            temp_mjd[passband] = [ex_set["mjd"][i]]
        else:
            temp_flux[passband].append(ex_set["flux"][i])
            temp_mjd[passband].append(ex_set["mjd"][i])
        if (i % 10000) == 0:
            print("line", i)
        i += 1
        if i == set_length-1:
            done = True
            break

    ids.append(current_id)
    mjd.append(temp_mjd)
    flux.append(temp_flux)



#print(mjd[0], flux[0])
# print(ids[0], flux[0], mjd[0])
poly_coefs = poly_extract(mjd, flux)
assert len(poly_coefs) == len(ids)
poly_data = zip(ids, poly_coefs)
# TO CSV:
# object_id, passband, coef_0, coef_1, coef_2 ... coef_n
f = open("polynomial_coefficients_train.csv", 'w')
f.write("object_id, passband, coef_0, coef_1, coef_2, coef_4\n")
for id, coef_group in poly_data:
    for passband in coef_group:
        coef_string = ""
        for coef in coef_group[passband]:
            coef_string += f"{coef},"
        coef_string = coef_string[:-1]
        f.write(f"{id},{passband},{coef_string}\n")
f.close()
print(poly_coefs[0][0])
plotter_poly = np.poly1d(poly_coefs[0][0])
print(plotter_poly)


times = []
fluxes = []
times_1 = []
fluxes_1 = []
target_id = 615
for i in range(len(ex_set['object_id'])):
    # I'm 100% sure there's some pandas bullshittery I can do here that works way better than this
    if ex_set['object_id'][i] > target_id:
            break
    if 0 == ex_set['passband'][i] and ex_set['object_id'][i] == target_id:
        times.append(ex_set['mjd'][i])
        fluxes.append(ex_set['flux'][i])

    if 1 == ex_set['passband'][i] and ex_set['object_id'][i] == target_id:
        times_1.append(ex_set['mjd'][i])
        fluxes_1.append(ex_set['flux'][i])

min_time = min(times)
corrected_times = np.subtract(times, min_time)
corrected_times_1 = np.subtract(times_1, min_time)
plotter_poly = np.poly1d(poly_coefs[0][0])
plotter_poly_1 = np.poly1d(poly_coefs[0][1])
output_poly = plotter_poly(range(800))
output_poly_1 = plotter_poly_1(range(800))

plt.scatter(corrected_times, fluxes)
plt.plot(range(800), output_poly)

plt.scatter(corrected_times_1, fluxes_1)
plt.plot(range(800), output_poly_1)
plt.show()
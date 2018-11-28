import pandas as pd
import matplotlib.pyplot as plt


times = []
fluxes = []
times_2 = []
fluxes_2 = []
flux_err_upper = []
flux_err_lower = []
test_set = pd.read_csv('training_set.csv')
target_id = 713

for i in range(len(test_set['object_id'])):
    if test_set['object_id'][i] > target_id:
            break
    if 0 < test_set['mjd'][i] and test_set['object_id'][i] == target_id:
        if test_set['detected'][i]:
            times.append(test_set['mjd'][i])
            fluxes.append(test_set['flux'][i])
            flux_err_lower.append(test_set['flux'][i] - test_set['flux_err'][i])
            flux_err_upper.append(test_set['flux'][i] + test_set['flux_err'][i])
        else:
            times_2.append(test_set['mjd'][i])
            fluxes_2.append(test_set['flux'][i])

print(times[0], fluxes[0], flux_err_upper[0], flux_err_lower[0], test_set['flux_err'][0])

metadata = pd.read_csv('training_set_metadata.csv')

print("Beginning catastrophic redshift calc")
total_objects = len(metadata['object_id'])
non_cat = 0
super_cat = 0
for i in range(len(metadata['object_id'])):
    upper_bound = metadata['hostgal_photoz'][i] + metadata['hostgal_photoz_err'][i]
    lower_bound = metadata['hostgal_photoz'][i] - metadata['hostgal_photoz_err'][i]
    if not metadata['hostgal_specz'][i]:
        total_objects -= 1
    elif lower_bound <= metadata['hostgal_specz'][i] <= upper_bound:
        non_cat += 1
    elif (upper_bound + 2*metadata['hostgal_photoz_err'][i]) < metadata['hostgal_specz'][i] or \
            (lower_bound - 2*metadata['hostgal_photoz_err'][i]) > metadata['hostgal_specz'][i]:
        super_cat += 1

    if metadata['object_id'][i] == target_id:
        print(metadata['target'][i])

print("Number of Out of Error Bounds redshifts (discounting no data):", total_objects - non_cat)
print("Number of Catastrophic Redshifts:", super_cat)
print("Number of objects", total_objects)
print("% Out of Bounds", (total_objects-non_cat)/total_objects)
print("% Catastrophic", super_cat/total_objects)

plt.scatter(times, fluxes)
plt.scatter(times_2, fluxes_2)
plt.show()
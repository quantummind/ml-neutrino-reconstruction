#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
alpha=0.05


# In[80]:


directory = 'pi0-final/nn-corrected-cone-2/'

# fitted_gamma0_mmgu = np.load(directory + 'pi0_fitted_gamma0_mmgu.npy')
# fitted_gamma1_mmgu = np.load(directory + 'pi0_fitted_gamma1_mmgu.npy')
# true_voxels_gamma0_mmgu = np.load(directory + 'pi0_true_voxels_gamma0_mmgu.npy')
# true_voxels_gamma1_mmgu = np.load(directory + 'pi0_true_voxels_gamma1_mmgu.npy')

e1 = np.load(directory + 'pi0_true_energy_fitted_voxels.npy').astype(np.float64)
e2 = np.load(directory + 'pi0_charge_fitted_voxels.npy').astype(np.float64)
e3 = np.load(directory + 'pi0_fitted_energy_fitted_voxels.npy').astype(np.float64)
e4 = np.load(directory + 'pi0_fitted_energy_recorded_voxels.npy').astype(np.float64)
e5 = np.load(directory + 'pi0_charge_recorded_voxels.npy').astype(np.float64)
e6 = np.load(directory + 'pi0_true_energy_recorded_voxels.npy').astype(np.float64)
e7 = np.load(directory + 'pi0_true_energy_all_voxels.npy').astype(np.float64)


true_gamma0 = np.load(directory + 'pi0_true_gamma0.npy')
true_gamma1 = np.load(directory + 'pi0_true_gamma1.npy')
fitted_cos = np.load(directory + 'pi0_fitted_cos.npy')
fitted_angle = np.arccos(np.clip(fitted_cos,-0.9999999999,0.9999999999))*180/np.pi
true_cos = np.load(directory + 'pi0_true_cos.npy', allow_pickle=True)
true_angle = np.arccos(np.clip(true_cos,-0.9999999999,0.9999999999))*180/np.pi
gamma_sep = np.load(directory + 'pi0_gamma_sep.npy')
gamma_dir = np.load(directory + 'pi0_gamma_dir.npy')
gamma_pca = np.load(directory + 'pi0_gamma_pca.npy', allow_pickle=True)
gamma_pca_nhits = np.load(directory + 'pi0_gamma_pca_nhits.npy')


# In[81]:


print('total number of pi0s reconstructed', len(e1))


# In[62]:


angle_selection = fitted_cos > 0.7
low_gamma_selection_0 = np.minimum(e3[:, 0], e3[:, 1]) < 80
low_gamma_selection_1 = np.maximum(e3[:, 0], e3[:, 1]) < 170
equiv_gamma_selection = e3[:, 0] != e3[:, 1]

selection = angle_selection & low_gamma_selection_0 & low_gamma_selection_1
# selection = np.arange(len(fitted_cos))
print('number of pi0s within clip on angle and energy', np.sum(selection))


# In[63]:


errors = np.abs(fitted_cos - true_cos)
bins = np.linspace(-1, 1, num=20)
inds = np.digitize(fitted_cos, bins)


# In[64]:


for i in range(1, len(bins)):
    error = np.mean(errors[np.where(inds == i)])
    print(bins[i], error)


# In[65]:


print(len(np.where(fitted_cos > 0.7)[0])/len(fitted_cos))


# In[67]:


get_ipython().run_line_magic('matplotlib', 'notebook')

plt.figure()
plt.scatter(fitted_cos, true_cos, s=1)
plt.show()


# In[52]:


errors = np.abs(np.sqrt(e3[:, 0]*e3[:, 1]) - np.sqrt(e6[:, 0]*e6[:, 1])*np.sqrt(2*(1-true_cos)))
bins = np.linspace(0, 500, num=20)
inds = np.digitize(np.minimum(e3[:, 1], e3[:, 0]), bins)
for i in range(1, len(bins)):
    error = np.median(errors[np.where(inds == i)])
    print(bins[i], error)


# In[70]:


errors = np.abs(np.sqrt(e3[:, 0]*e3[:, 1]) - np.sqrt(e6[:, 0]*e6[:, 1])*np.sqrt(2*(1-true_cos)))
bins = np.linspace(0, 500, num=20)
inds = np.digitize(np.maximum(e3[:, 1], e3[:, 0]), bins)
for i in range(1, len(bins)):
    error = np.median(errors[np.where(inds == i)])
    print(bins[i], error)


# In[72]:


ir_charge_cone = IsotonicRegression(out_of_bounds='clip').fit(e2[selection].flatten(), e7[selection].flatten())
ir_energy_cone = IsotonicRegression(out_of_bounds='clip').fit(e3[selection].flatten(), e7[selection].flatten())
print(r2_score(ir_charge_cone.predict(e2[selection].flatten()), e7[selection].flatten()))
print(r2_score(ir_energy_cone.predict(e3[selection].flatten()), e7[selection].flatten()))


# In[73]:


lr_charge_cone = LinearRegression().fit(e2[selection].flatten().reshape(-1, 1), e7[selection].flatten())
lr_energy_cone = LinearRegression().fit(e3[selection].flatten().reshape(-1, 1), e7[selection].flatten())
print(r2_score(lr_charge_cone.predict(e2[selection].flatten().reshape(-1, 1)), e7[selection].flatten()))
print(r2_score(lr_energy_cone.predict(e3[selection].flatten().reshape(-1, 1)), e7[selection].flatten()))


# In[74]:


lr_charge = LinearRegression().fit(e5.flatten().reshape(-1, 1), e7.flatten())
lr_energy = LinearRegression().fit(e4.flatten().reshape(-1, 1), e7.flatten())
print(r2_score(lr_charge.predict(e5.flatten().reshape(-1, 1)), e7.flatten()))
print(r2_score(lr_energy.predict(e4.flatten().reshape(-1, 1)), e7.flatten()))


# In[75]:


import pickle

# mmgu_factor = 0.0156 * 1.27  # correction from true voxel + true angle estimate
# fitted_gamma0 = mmgu_factor * fitted_gamma0_mmgu
# fitted_gamma1 = mmgu_factor * fitted_gamma1_mmgu
# tv_gamma0 = mmgu_factor * true_voxels_gamma0_mmgu
# tv_gamma1 = mmgu_factor * true_voxels_gamma1_mmgu
# fa_fm = np.sqrt(2 * (fitted_gamma0 * fitted_gamma1) * (1 - fitted_cos))
# fa_tm = np.sqrt(2 * (true_gamma0 * true_gamma1) * (1 - fitted_cos))
# ta_tm = np.sqrt(2 * (true_gamma0 * true_gamma1) * (1 - true_cos))
# ta_fm = np.sqrt(2 * (fitted_gamma0 * fitted_gamma1) * (1 - true_cos))
# ta_tvm = np.sqrt(2 * (tv_gamma0 * tv_gamma1) * (1 - true_cos))
# fa_tvm = np.sqrt(2 * (tv_gamma0 * tv_gamma1) * (1 - fitted_cos))

# TODO fix RHS for e1-e3

def measure_clusterer(c1, c2, c3, ind):
    c_filter = np.arange(len(c1))
#     c_filter = np.where(c1[:, ind, 0] != c1[:, ind, 1])[0]
    a1 = np.sqrt(2 * (c1[c_filter, 0] * c1[c_filter, 1]) * (1 - fitted_cos[c_filter]))
    a2 = np.sqrt(2 * (c2[c_filter, 0] * c2[c_filter, 1]) * (1 - fitted_cos[c_filter]))
    a3 = np.sqrt(2 * (c3[c_filter, 0] * c3[c_filter, 1]) * (1 - fitted_cos[c_filter]))
    a4 = np.sqrt(2 * (c3[c_filter, 0] * c3[c_filter, 1]) * (1 - true_cos[c_filter]))
    return a1, a2, a3, a4

cone_true_energy_fitted_voxels__fitted_angle, cone_charge_fitted_voxels__fitted_angle, cone_fitted_energy_fitted_voxels__fitted_angle, cone_fitted_energy_fitted_voxels__true_angle = measure_clusterer(e1, e2, e3, 0)
fit_cone = np.sqrt(2 * (ir_energy_cone.predict(e3[:, 0]) * ir_energy_cone.predict(e3[:, 1])) * (1 - fitted_cos))   # BEST YOU CAN DO
true_angle_fit_cone = np.sqrt(2 * (ir_energy_cone.predict(e3[:, 0]) * ir_energy_cone.predict(e3[:, 1])) * (1 - true_cos))   # BEST YOU CAN DO


fitted_energy_true_recorded_voxels__fitted_angle = np.sqrt(2 * (e4[:, 0] * e4[:, 1]) * (1 - fitted_cos))
fitted_energy_true_recorded_voxels__true_angle = np.sqrt(2 * (e4[:, 0] * e4[:, 1]) * (1 - true_cos))
charge_true_recorded_voxels__fitted_angle = np.sqrt(2 * (e5[:, 0] * e5[:, 1]) * (1 - fitted_cos))
charge_true_recorded_voxels__true_angle = np.sqrt(2 * (e5[:, 0] * e5[:, 1]) * (1 - true_cos))
true_energy_true_recorded_voxels__fitted_angle = np.sqrt(2 * (e6[:, 0] * e6[:, 1]) * (1 - fitted_cos))
true_energy_true_recorded_voxels__true_angle = np.sqrt(2 * (e6[:, 0] * e6[:, 1]) * (1 - true_cos))
true_energy_true_all_voxels__fitted_angle = np.sqrt(2 * (e7[:, 0] * e7[:, 1]) * (1 - fitted_cos))
true_energy_true_all_voxels__true_angle = np.sqrt(2 * (e7[:, 0] * e7[:, 1]) * (1 - true_cos))   # BEST YOU CAN DO
fit_true_energy_true_all_voxels__true_angle = np.sqrt(2 * (lr_energy.predict(e4[:, 0].reshape(-1,1)) * lr_energy.predict(e4[:, 1].reshape(-1,1))) * (1 - true_cos))   # BEST YOU CAN DO
charge_fit_true_energy_true_all_voxels__true_angle = np.sqrt(2 * (lr_charge.predict(e5[:, 0].reshape(-1,1)) * lr_charge.predict(e5[:, 1].reshape(-1,1))) * (1 - true_cos))   # BEST YOU CAN DO


# In[79]:


# np.savetxt('pi0-final/reco-true.txt', true_energy_true_all_voxels__true_angle)
# np.savetxt('pi0-final/reco-energy.txt', fit_true_energy_true_all_voxels__true_angle)
# np.savetxt('pi0-final/reco-charge.txt', charge_fit_true_energy_true_all_voxels__true_angle)
# np.savetxt('pi0-final/reco-cone-clipped.txt', fit_cone[selection])


# In[78]:


# selection = angle_selection & equiv_gamma_selection
# selection = np.arange(len(e3))
# selection = angle_selection


plt.figure()
plt.title('Cone clustering with U-ResNet energy')
plt.xlabel('Mass (MeV)')
plt.hist(cone_fitted_energy_fitted_voxels__fitted_angle[selection], range=(0, 1000), bins=100)

plt.figure()
plt.title('Cone clustering with true energy')
plt.xlabel('Mass (MeV)')
plt.hist(cone_true_energy_fitted_voxels__fitted_angle[selection], range=(0, 1000), bins=100)

plt.figure()
plt.title('Cone clustering pi0 mass (missing energy fitted)')
plt.xlabel('Mass (MeV)')
plt.hist(fit_cone[selection], range=(0, 1000), bins=80)

# plt.figure()
# plt.title('Cone clustering + U-ResNet energy + true angle')
# plt.xlabel('Mass (MeV)')
# plt.hist(true_angle_fit_cone[selection], range=(0, 1000), bins=100)




# plt.figure()
# plt.title('True voxels')
# plt.xlabel('Mass (MeV)')
# plt.hist(fitted_energy_true_recorded_voxels__fitted_angle[selection], range=(0, 1000), bins=100)

# plt.figure()
# plt.title('True angle')
# plt.xlabel('Mass (MeV)')
# plt.hist(fitted_energy_true_recorded_voxels__true_angle[selection], range=(0, 1000), bins=100)

# plt.figure()
# plt.title('If you used charge')
# plt.xlabel('Mass (MeV)')
# plt.hist(charge_true_recorded_voxels__true_angle[selection]*0.0156, range=(0, 1000), bins=100)

# plt.figure()
# plt.title('True energy')
# plt.xlabel('Mass (MeV)')
# plt.hist(true_energy_true_recorded_voxels__true_angle[selection], range=(0, 1000), bins=100)

# plt.figure()
# plt.title('Ground truth')
# plt.xlabel('Mass (MeV)')
# plt.hist(true_energy_true_all_voxels__true_angle[selection], range=(0, 1000), bins=100)

# plt.figure()
# plt.title('Energy fit')
# plt.xlabel('Mass (MeV)')
# plt.hist(fit_true_energy_true_all_voxels__true_angle[selection], range=(0, 1000), bins=100)

# plt.figure()
# plt.title('Charge fit')
# plt.xlabel('Mass (MeV)')
# plt.hist(charge_fit_true_energy_true_all_voxels__true_angle[selection], range=(0, 1000), bins=100)

plt.show()
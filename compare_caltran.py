import pandas as pd
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plot
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import norm
import caltran_utils as ct
import scipy as sp

# This function creates a plot to plot spectra on the same set of wavelengths.
# The assumption is that the spectra are: lab data, mars data, and a list of transformed lab spectra
# A list of labels can be provided along with the transformed lab spectra
#
def do_comparison_plot(wvls,lab,mars,transformed, transformed_label=[''],filename=''):
    plot.plot(wvls, lab, label='Lab', linewidth=0.5)
    for i in range(len(transformed)):
        plot.plot(wvls, transformed[i], label=transformed_label[i], linewidth=0.5)
    plot.plot(wvls, mars, label='Mars', linewidth=0.5)
    plot.legend()
    plot.savefig(filename)
    plot.close()

# Simple function to calculate the root mean squared error between two spectra,
# as an assement of how successful the tranformation was
def mismatch_rmse(spectrum, spectrum_to_match):
    return np.sqrt(np.average((spectrum - spectrum_to_match)**2))


#Begin code provided by T. Boucher, with minor tweaks ##############
def pls_ds(A,B,n_components=1):
    model = PLSRegression(n_components=n_components,scale=False).fit(B,A)
    return model

def direct_standardization(A,B,fit_intercept=False):
    assert A.shape[0]==B.shape[0], (
        'Input matrices must have the same number of rows (i.e. samples).')
    if fit_intercept:
        working_B = np.hstack((B,np.ones((B.shape[0],1))))
    else:
        working_B = np.copy(B)
    proj_to_A = np.dot(np.linalg.pinv(working_B),A)
    proj_B = np.dot(working_B,proj_to_A)
    return proj_to_A,proj_B

def piecewise_ds(A,B,win_size=5,pls=None):
    assert A.shape==B.shape, "Input matrices must be the same shape."
    assert win_size % 2 == 1, "Window size must be odd."

    padding = (win_size - 1) / 2
    n_feats = A.shape[1]

    coefs = []
    for i in range(n_feats):
        row = np.zeros(n_feats)
        start = int(max(i-padding,0))
        end = int(min(i+padding,n_feats-1)+1)
        if isinstance(pls,int):
            model = PLSRegression(n_components=pls,scale=False)
            try:
                model.fit(B[:,start:end],A[:,i])
                row[start:end] = model.coef_.ravel()
            except:
                pass
        elif pls is None:
            row[start:end]=np.dot(np.linalg.pinv(B[:,start:end]),A[:,i])
        else:
            print("ERROR: bad number of PLS components.")
            return
        coefs.append(row)

    proj_to_A = np.array(coefs).T
    proj_B = np.dot(B,proj_to_A)

    return proj_to_A,proj_B

def lasso_ds(A,B,rho=1,beta=.02):
    return admm_ds(A,B,rho=rho,beta=beta,max_iter=100,reg='lasso')

def ridge_ds(A,B,rho=1,beta=.02):
    return admm_ds(A,B,rho=rho,beta=beta,max_iter=100,reg='ridge')


def admm_ds(A,B,rho=1,beta=.02,epsilon=1e-5,max_iter=100,verbose=True,
            reg='lasso'):

    n = B.shape[1]
    P = np.zeros((n,n))
    Z = P.copy()
    Y = P.copy()

    A = normalize(A, axis=1)
    B = normalize(B, axis=1)

    BtB=np.dot(B.T,B)
    BtA=np.dot(B.T,A)
    if reg is 'fused':
        I = np.identity(n-1)
        pos = np.hstack((I,np.zeros((len(I),1))))
        neg = -1*np.roll(pos,1,1)
        D = np.vstack(((pos+neg),np.zeros((1,len(I)+1))))
        P_fact = cho_factor(BtB + rho * np.dot(D.T,D))
    else:
        P_fact = cho_factor(BtB + rho * np.eye(n))
    if reg is 'ridge':
        Z_fact = cho_factor(2*np.eye(n) + rho*np.eye(n))

    for it in range(max_iter):
        last_P = P
        last_Z = Z
        if reg is 'fused':
            Z = ct.soft_thresh(np.dot(D,P)+(Y/rho), beta/rho)
        elif reg is 'rank':
            Z = ct.svt_thresh(P+(Y/rho), beta/rho)
        elif reg is 'ridge':
            Z = cho_solve(Z_fact, rho*P + Y)
        elif reg is 'sp_lr':
            Z = (ct.soft_thresh(P+(Y/rho),beta/rho) +
                 ct.svt_thresh(P+(Y/rho),beta/rho)) / 2.0
        elif reg is 'lasso':
            Z = ct.soft_thresh(P+(Y/rho), beta/rho)
        else:
            return 'ERROR: Regularizer not programmed.'
        if reg is 'fused':
            P = cho_solve(P_fact, BtA + rho*np.dot(D.T,Z) - np.dot(D.T,Y))
            Y += rho*(np.dot(D,P) - Z)
        else:
            P = cho_solve(P_fact, BtA + rho*Z - Y)
            Y += rho*(P - Z)
        P_conv = norm(P-last_P) / norm(P)
        Z_conv = norm(Z-last_Z) / norm(Z)
        if verbose:
            # num_zero = np.count_nonzero(P<=1e-9)
            if reg is 'fused':
                print (it, P_conv, Z_conv, norm(np.dot(D,P)-Z), np.count_nonzero(Z), sum(sp.linalg.svdvals(Z)))
                print ("score: %.4f" % (norm(np.dot(D,P)-Z)+norm(A-B.dot(Z))))
            else:
                print (it, P_conv, Z_conv, norm(P-Z), np.count_nonzero(Z), norm(Z,1))  # ,sum(sp.linalg.svdvals(Z)))
                print ("score: %.4f" % (norm(P-Z)+norm(A-B.dot(Z))))
        if P_conv <= epsilon and Z_conv <= epsilon:
            break
    else:
        print ("Didn't converge in %d steps" % max_iter)

    return Z,np.dot(B,P)

def sparse_lowrank_ds(A,B):
    return admm_ds(A,B,reg='sp_lr',rho=1,beta=.02)

def cca_ds(A,B,n_components=1):
    model = CCA(n_components=n_components,scale=False).fit(B,A)
    return model


def new_cca_ds(A,B,n_components=1):
    # http://onlinelibrary.wiley.com/doi/10.1002/cem.2637/abstract
    model = CCA(n_components=n_components,scale=False).fit(B,A)
    F1 = np.linalg.pinv(model.x_scores_).dot(model.y_scores_)
    F2 = np.linalg.pinv(model.y_scores_).dot(A)
    P = ct.multi_dot((model.x_weights_, F1, F2))
    return P, B.dot(P)

def incr_prox_descent_ds(A,B,t=.0002,svt=10,l1=10,epsilon=1e-5,max_iter=50,
                         verbose=True):
    # incremental proximal descent, Bertsekas 2010
    P = np.eye(B.shape[1])
    #P = np.zeros(B.shape[1])
    A = normalize(A, axis=1)
    B = normalize(B, axis=1)

    BtB=np.dot(B.T,B)
    BtA=np.dot(B.T,A)
    for it in range(max_iter):
        last_P = P.copy()
        P = P - t*(np.dot(BtB,P)-BtA)
        P = ct.svt_thresh(P, svt*t)
        P = ct.soft_thresh(P, l1*t)
        P_conv = norm(P-last_P) / norm(P)
        if verbose:
            #svdsum = sum(sp.linalg.svdvals(P))
            #print(it, P_conv, norm(A-B.dot(P)), norm(P,1), svdsum)
            print(it, P_conv, norm(A-B.dot(P)), norm(P,1))
            # print("score: %.4f" % (norm(A-B.dot(P))+norm(P,1)+svdsum))
        if P_conv <= epsilon:
            break
    else:
        print("Didn't converge in %d steps" % max_iter)
    return P,np.dot(B,P)

def forward_backward_ds(A,B,t=0.001,svt=1,l1=1,epsilon=1e-5,max_iter=20,
                        verbose=True):
    #P = np.eye(B.shape[1])
    P = np.zeros(B.shape[1])
    Z1 = P.copy()
    Z2 = P.copy()

    A = normalize(A, axis=1)
    B = normalize(B, axis=1)

    BtB=np.dot(B.T,B)
    BtA=np.dot(B.T,A)
    for it in range(max_iter):
        last_P = P.copy()
        G = np.dot(BtB,P)-BtA
        Z1 = ct.svt_thresh(2*P-Z1-t*G, 2*svt*t)
        Z2 = ct.soft_thresh(2*P-Z2-t*G, 2*l1*t)
        P = (Z1+Z2) / 2.0
        P_conv = norm(P-last_P) / norm(P)
        if verbose:
            print(it, P_conv)
        if P_conv <= epsilon:
            break
    else:
        print("Didn't converge in %d steps" % max_iter)

    return P,np.dot(B,P)

#################################################
# Main script for evaluating caltran methods.
# Two different methods of identifying corresponding spectra on Earth and Mars have been tried: Means and "Closest"
# For the "means" method, we calculate the average spectrum of each target on earth and match it up with the average
# spectrum of each target on Mars.
# For the "closest" method, use line ratios to identify individual spectra from earth and Mars that are most similar
# and use these instead of the mean spectra

#Load Lab cal target data
lab_data = pd.read_csv(
    r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\caltran\lab_data_cal_targets_means.csv",
    header=[0,1])
#lab_data = pd.read_csv(
# r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\caltran\lab_cal_targets_closest_to_mars.csv",
#  header=[0,1])

#Code to mean center the lab spectra
#TODO: Does mean centering help at all?
#lab_mean = np.mean(lab_data['wvl'],axis=0)
#lab_data['wvl']=lab_data['wvl'].apply(lambda x: x-x.mean())


#Load Mars Cal target data
mars_data = pd.read_csv(
    r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\caltran\mars_cal_targets_means.csv",
    header=[0,1])
#mars_data = pd.read_csv(
# r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\caltran\mars_cal_targets_closest_toLab.csv",
# header=[0,1])

#Code to mean center the lab spectra
#TODO: Does mean centering help at all?
# mars_mean = np.mean(mars_data['wvl'],axis=0)
# mars_data['wvl']=mars_data['wvl'].apply(lambda x: x-x.mean())

#Specify in the output files which spectra are being used
#outname = '_closest_'
outname='_mean_'

# load earth to mars correction currently being used
# (This is just a simple vector derived from the ratio of the earth and Mars data)
e2m = pd.read_csv(
    r"C:\Users\rbanderson\Documents\Projects\MSL\ChemCam\DataProcessing\Working\caltran\EARTH_2_MARS_CORR.CSV",
    header=None)

#iterate through each cal target, run each cal tran method, calculate spectral mismatch, save transformed spectra
cal_targets_unique = lab_data[('meta','Target')] #get list of cal targets
wvls = np.array(lab_data['wvl'].columns.values,dtype=float) #get wvls

#do CV to compare results
cv_results = pd.DataFrame() #make an empty data frame to hold the results
ind = 0

#for each cal target
for target in cal_targets_unique:
    #split mars and lab data into training and validation
    val_data_lab = np.squeeze(np.array(lab_data[lab_data[('meta','Target')] == target]['wvl']))
    train_data = np.squeeze(np.array(lab_data[lab_data[('meta', 'Target')] != target]['wvl']))
    val_data_mars = np.squeeze(np.array(mars_data[mars_data[('meta', 'Target_x')] == target]['wvl']))
    train_data_mars = np.squeeze(np.array(mars_data[mars_data[('meta', 'Target_x')] != target]['wvl']))

    #This is some code I was fiddling with to try to get the decomposition to work
    # #Identify and remove zeros
    # BtB_sum = np.sum(np.dot(train_data.T,train_data),axis=0)
    # train_data=train_data[:,BtB_sum!=0]
    # val_data_lab = val_data_lab[BtB_sum!=0]
    # train_data_mars = train_data_mars[:,BtB_sum!=0]
    # val_data_mars = val_data_mars[BtB_sum!=0]
    # e2m=e2m.iloc[BtB_sum!=0]
    # wvls = wvls[BtB_sum!=0]

   # This code was used to make plots of the error in the transform for piecewise_ds (the best method so far)
   #   # make plots
   #  pds_do_proj, train_pds_proj = piecewise_ds(train_data_mars, train_data, win_size=1, pls=None)
   #  val_data_lab_pds1 = np.dot(val_data_lab, pds_do_proj)
   #  val_data_lab_ratio = np.squeeze(np.array(val_data_lab * e2m[1]))
   # # plot.plot(wvls, val_data_lab-val_data_mars, label='Uncorrected', linewidth=0.5)
   #  plot.plot(wvls, abs(val_data_lab_ratio-val_data_mars), label='Ratio', linewidth=0.5)
   #  plot.plot(wvls, abs(val_data_lab_pds1-val_data_mars), label='PDS (window=1)', linewidth=0.5)
   #  plot.title(target+' Lab to Mars absolute difference')
   #  plot.legend()
   #  plot.savefig(target+outname+'compare_ratio_pds1.png',dpi=800)
   #  plot.close()
   #
   #
   #  plot.plot(wvls,e2m[1],label='Ratio',linewidth=0.5)
   #  plot.plot(wvls,val_data_lab_pds1/val_data_lab,label='PDS (width=1)',linewidth=0.5)
   #  plot.title(target+' transformation vectors')
   #  plot.legend()
   #  plot.ylim([-5,25])
   #  plot.savefig(target+outname+'compare_transform_vect.png',dpi=800)
   #  plot.close()
   #  pass

    print("Calculating results with No transformation applied")
    cv_results.loc[ind,'Method']='None'
    cv_results.loc[ind,target+'_RMSE']= mismatch_rmse(val_data_lab,val_data_mars)
    ind=ind+1

    print("Calculating results with current ratio method")
    val_data_lab_transformed =np.squeeze(np.array(val_data_lab*e2m[1]))
    cv_results.loc[ind,'Method']='Ratio'
    cv_results.loc[ind,target+'_RMSE']=mismatch_rmse(val_data_lab_transformed,val_data_mars)
    ind=ind+1

    do_comparison_plot(wvls,val_data_lab,val_data_mars,[val_data_lab_transformed],
                           transformed_label=['Lab (Ratio)'],
                           filename=target+outname+'_ratio.png')

    print("Calculating results using Ridge DS")
    Z, train_data_transformed = ridge_ds(train_data_mars, train_data, rho=1, beta=.02)

    cv_results.loc[ind, 'Method'] = 'Ridge DS'
    cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
    ind = ind + 1
    pass

    print("Calculating results using Direct standardization")
    ds_do_proj, train_ds_proj = direct_standardization(train_data_mars, train_data, fit_intercept=False)
    val_data_lab_transformed = np.dot(val_data_lab, ds_do_proj)
    cv_results.loc[ind, 'Method'] = 'DS'
    cv_results.loc[ind, 'fit_intercept'] = 'False'
    cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
    ind = ind + 1
    do_comparison_plot(wvls, val_data_lab, val_data_mars, [val_data_lab_transformed],
                       transformed_label=['Lab (DS fit_int=False)'],
                       filename=target + outname + '_ds.png')

    print("Calculating results using Piecewise direct standardization")
    for win_size in range(13, 20, 2):
        print('Window size: '+str(win_size))
        pds_do_proj, train_pds_proj = piecewise_ds(train_data_mars, train_data, win_size=win_size, pls=None)
        val_data_lab_transformed = np.dot(val_data_lab, pds_do_proj)
        cv_results.loc[ind, 'Method'] = 'PDS'
        cv_results.loc[ind, 'window'] = win_size
        cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
        print("RMSE = "+str(mismatch_rmse(val_data_lab_transformed, val_data_mars)))
        ind = ind + 1
        #I forget why this is commented out. It either wasn't running, or was running too slowly...

        # for nc in range(1,min(10, win_size-1)):
        #     print("PDS-PLS nc="+str(nc))
        #     pds_do_proj, train_pds_proj = piecewise_ds(train_data_mars, train_data, win_size=win_size, pls=nc)
        #     val_data_lab_transformed = np.dot(val_data_lab, pds_do_proj)
        #     cv_results.loc[ind, 'Method'] = 'PDS-PLS'
        #     cv_results.loc[ind, 'window'] = win_size
        #     cv_results.loc[ind, 'nc'] = nc
        #     cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
        #     print("RMSE = " + str(mismatch_rmse(val_data_lab_transformed, val_data_mars)))
        #     ind = ind + 1

    print("Calculating results using incr Prox descent")
    do_ipd,train_transformed = incr_prox_descent_ds(train_data_mars,train_data,t=.00000002,svt=10,l1=10,epsilon=1e-5,max_iter=50,
                         verbose=True)
    val_data_lab_transformed = np.squeeze(np.array(np.dot(val_data_lab,do_ipd)))
    cv_results.loc[ind, 'Method'] = 'IPD'
    cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
    ind=ind+1

    print("Calculating results using forward backward ds")
    do_fbds,train_transformed = forward_backward_ds(train_data_mars,train_data,t=.0001,svt=1,l1=1,epsilon=1e-5,max_iter=20,
                         verbose=True)
    val_data_lab_transformed = np.squeeze(np.array(np.dot(val_data_lab,do_fbds)))
    cv_results.loc[ind, 'Method'] = 'FBDS'
    cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
    ind=ind+1

    print("Calculating results using PLS Direct Standardization")
    for nc in range(1,10):
        print('nc = '+str(nc))
        pls_ds_model = pls_ds(train_data_mars, train_data, n_components=nc)
        val_data_lab_transformed = np.squeeze(np.array(pls_ds_model.predict(val_data_lab.reshape(1,-1))))
        cv_results.loc[ind,'Method']='PLS-DS'
        cv_results.loc[ind,'nc']=nc
        cv_results.loc[ind,target+'_RMSE']=mismatch_rmse(val_data_lab_transformed,val_data_mars)
        ind=ind+1
        do_comparison_plot(wvls,val_data_lab,val_data_mars,[val_data_lab_transformed],
                           transformed_label=['Lab (PLS-DS nc=' + str(nc) + ')'],
                           filename=target+outname+'_pls_ds_nc'+str(nc)+'.png')


    print("Calculating results using CCA-DS")
    for nc in range(1,10):
        cca_ds_model = cca_ds(train_data_mars, train_data, n_components=nc)
        val_data_lab_transformed = cca_ds_model.predict(val_data_lab.reshape(1,-1))
        cv_results.loc[ind, 'Method'] = 'CCA-DS'
        cv_results.loc[ind, 'nc'] = nc
        cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
        ind = ind + 1

    print("Calculating results using New CCA-DS")
    for nc in range(1,10):
        do_new_ccs_ds,train_cca_trans = new_cca_ds(train_data_mars, train_data, n_components=nc)
        val_data_lab_transformed = val_data_lab.dot(do_new_ccs_ds)
        cv_results.loc[ind, 'Method'] = 'New CCA-DS'
        cv_results.loc[ind, 'nc'] = nc
        cv_results.loc[ind, target + '_RMSE'] = mismatch_rmse(val_data_lab_transformed, val_data_mars)
        ind = ind + 1

    ind=0 #reset the index for next time
cv_results.to_csv(outname+'cv_results.csv') #write the cross validation results out to a csv file

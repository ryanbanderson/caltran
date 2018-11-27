from collections import defaultdict
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

import cProfile
import os.path
import pstats
import StringIO
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


import caltran_utils as ct

def load_noise(n_samples=100,n_feats=20,k=500):
    A = np.random.rand(n_samples,n_feats)
    B = np.copy(A) + np.random.randn(n_samples,n_feats)*.01
    return [A,B]


def load_corn(base_dir='~/Mars/Data/Transfer/corn/'):
    base_dir = os.path.expanduser(base_dir)
    labels = np.genfromtxt(base_dir+'labels.csv',delimiter=',')
    spectra = [np.genfromtxt(base_dir+'m5spectra.csv',delimiter=','),
               np.genfromtxt(base_dir+'mp5spectra.csv',delimiter=','),
               np.genfromtxt(base_dir+'mp6spectra.csv',delimiter=',')]
    return spectra,labels


def load_shootout(base_dir='~/Mars/Data/Transfer/shootout/',
                  dataset='calibrate'):
    base_dir = os.path.expanduser(base_dir)
    if dataset in ['calibrate','validate','test']:
        labels = np.genfromtxt(base_dir+'%s_labels.csv' % dataset,delimiter=',')
    spectra = [np.genfromtxt(base_dir+'%s1.csv' % dataset,delimiter=','),
               np.genfromtxt(base_dir+'%s2.csv' % dataset,delimiter=',')]
    return spectra,labels


def sams_method(A,B,max_ratio=5):
    ratios = (A/B).mean(0)
    ratios = ratios.reshape(1,len(ratios))
    ratios = np.sign(ratios) * np.minimum(np.abs(ratios),max_ratio)
    proj_to_A = np.repeat(ratios,B.shape[1],axis=0).T
    return proj_to_A, np.dot(B,proj_to_A)


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
    for i in xrange(n_feats):
        row = np.zeros(n_feats)
        start = max(i-padding,0)
        end = min(i+padding,n_feats-1)+1
        if isinstance(pls,int):
            model = PLSRegression(n_components=pls,scale=False)
            model.fit(B[:,start:end],A[:,i])
            row[start:end]=model.coefs.ravel()
        elif pls is None:
            row[start:end]=np.dot(np.linalg.pinv(B[:,start:end]),A[:,i])
        else:
            print "ERROR: bad number of PLS components."
            return
        coefs.append(row)

    proj_to_A = np.array(coefs).T
    proj_B = np.dot(B,proj_to_A)

    return proj_to_A,proj_B


def sparse_lowrank_ds(A,B):
    #return forward_backward_ds(A,B,t=0.001,svt=1,l1=1,epsilon=1e-5,max_iter=20,verbose=True)
    #return incr_prox_descent_ds(A,B)
    return admm_ds(A,B,reg='sp_lr',rho=1,beta=.02)


def incr_prox_descent_ds(A,B,t=.0002,svt=10,l1=10,epsilon=1e-5,max_iter=50,
                         verbose=True):
    # incremental proximal descent, Bertsekas 2010
    #P = np.eye(B.shape[1])
    P = np.zeros(B.shape[1])
    BtB=np.dot(B.T,B)
    BtA=np.dot(B.T,A)
    for it in xrange(max_iter):
        last_P = P.copy()
        P = P - t*(np.dot(BtB,P)-BtA)
        P = ct.svt_thresh(P, svt*t)
        P = ct.soft_thresh(P, l1*t)
        P_conv = norm(P-last_P) / norm(P)
        if verbose:
            svdsum = sum(sp.linalg.svdvals(P))
            print it, P_conv, norm(A-B.dot(P)), norm(P,1), svdsum
            #print "score: %.4f" % (norm(A-B.dot(P))+norm(P,1)+svdsum)
        if P_conv <= epsilon:
            break
    else:
        print "Didn't converge in %d steps" % max_iter
    return P,np.dot(B,P)


def forward_backward_ds(A,B,t=0.001,svt=1,l1=1,epsilon=1e-5,max_iter=20,
                        verbose=True):
    #P = np.eye(B.shape[1])
    P = np.zeros(B.shape[1])
    Z1 = P.copy()
    Z2 = P.copy()
    BtB=np.dot(B.T,B)
    BtA=np.dot(B.T,A)
    for it in xrange(max_iter):
        last_P = P.copy()
        G = np.dot(BtB,P)-BtA
        Z1 = ct.svt_thresh(2*P-Z1-t*G, 2*svt*t)
        Z2 = ct.soft_thresh(2*P-Z2-t*G, 2*l1*t)
        P = (Z1+Z2) / 2.0
        P_conv = norm(P-last_P) / norm(P)
        if verbose:
            print it, P_conv
        if P_conv <= epsilon:
            break
    else:
        print "Didn't converge in %d steps" % max_iter

    return P,np.dot(B,P)


def pls_ds(A,B,n_components=1):
    model = PLSRegression(n_components=n_components,scale=False).fit(B,A)
    return model.coefs,model.predict(B)


def cca_ds(A,B,n_components=1):
    model = CCA(n_components=n_components,scale=False).fit(B,A)
    return model.coefs,model.predict(B)


def new_cca_ds(A,B,n_components=1):
    # http://onlinelibrary.wiley.com/doi/10.1002/cem.2637/abstract
    model = CCA(n_components=n_components,scale=False).fit(B,A)
    F1 = np.linalg.pinv(model.x_scores_).dot(model.y_scores_)
    F2 = np.linalg.pinv(model.y_scores_).dot(A)
    P = ct.multi_dot((model.x_weights_, F1, F2))
    return P, B.dot(P)


def lowrank_ds(A,B):
    return admm_ds(A,B,rho=1,beta=1,reg='rank',max_iter=100)


def lasso_ds(A,B,rho=1,beta=.02):
    return admm_ds(A,B,rho=rho,beta=beta,max_iter=100,reg='lasso')


def fused_ds(A,B):
    return admm_ds(A,B,rho=1,beta=.01,epsilon=1e-5,max_iter=100,verbose=True,
                   reg='fused')


def ridge_ds(A,B,rho=1,beta=.5):
    return admm_ds(A,B,rho=rho,beta=beta,epsilon=1e-5,max_iter=100,verbose=True,
                   reg='ridge')


def admm_ds(A,B,rho=1,beta=.02,epsilon=1e-5,max_iter=100,verbose=False,
            reg='lasso'):
    n = B.shape[1]
    P = np.zeros((n,n))
    Z = P.copy()
    Y = P.copy()

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

    for it in xrange(max_iter):
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
                print it, P_conv, Z_conv, norm(np.dot(D,P)-Z),
                print np.count_nonzero(Z), sum(sp.linalg.svdvals(Z))
                print "score: %.4f" % (norm(np.dot(D,P)-Z)+norm(A-B.dot(Z)))
            else:
                print it, P_conv, Z_conv, norm(P-Z), np.count_nonzero(Z),
                print norm(Z,1)  # ,sum(sp.linalg.svdvals(Z))
                print "score: %.4f" % (norm(P-Z)+norm(A-B.dot(Z)))
        if P_conv <= epsilon and Z_conv <= epsilon:
            break
    else:
        print "Didn't converge in %d steps" % max_iter

    return Z,np.dot(B,P)


def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))


def run_full_caltarget_test(cal_spectra,cal_labels,cal_names,transformed_test,pls_comps=[10]):
    samples,comps = ct.load_data(norm=3,masked=True)
    org_samples = np.copy(samples)
    org_comps = np.copy(comps)

    elements = ['SiO2','TiO2','Al2O3','FeOT','MnO','MgO','CaO','Na2O','K2O']
    for n_comps in pls_comps:
        mars_preds = []
        ct_preds = []
        gt_preds = []
        for e,elem in enumerate(elements):
            if verbose:
                print '-----------------------'
                print elem
            for transformer in transformed_test.keys():
                print transformer
                targets = transformed_test[transformer]
                for t,target in enumerate(targets):
                    # Remove caltargets
                    if (comps['Name'] == cal_names[t]).any():
                        ind = np.argwhere(comps['Name']==cal_names[t])[0,0]
                        comps = np.delete(org_comps,ind,0)
                        samples = np.delete(org_samples,ind,0)
                    model = PLSRegression(n_components=n_comps,scale=False)
                    model.fit(samples,comps[elem])
                    lab_pred = model.predict(cal_spectra[0][t][None])
                    mars_pred = model.predict(cal_spectra[1][t][None])
                    trans_pred = model.predict(target)
                    gt = cal_labels[t,e]
                    score = (norm(mars_pred-gt,ord=1) -
                             norm(trans_pred-gt,ord=1))
                    mars_preds.append(mars_pred[0][0])
                    ct_preds.append(trans_pred[0][0])
                    gt_preds.append(gt)
                    if verbose:
                        print cal_names[t]
                        print 'Ground truth: %.4f' % gt
                        print 'Lab target: %.4f' % lab_pred
                        print 'Mars target: %.4f' % mars_pred
                        print 'Transformed Mars: %.4f' % trans_pred
                        print 'Score: %.4f' % score
                        print
        pred_shape = (len(elements),len(targets))
        ct_preds = np.array(ct_preds).reshape(pred_shape)
        gt_preds = np.array(gt_preds).reshape(pred_shape)
        mars_preds = np.array(mars_preds).reshape(pred_shape)
        print '-----------------------'
        print "Element\tMars\t\tCalTran\t\t%Gain/lost"
        for i,e in enumerate(elements):
            mars_rmsep = rmse(gt_preds[i,:],mars_preds[i,:])
            ct_rmsep = rmse(gt_preds[i,:],ct_preds[i,:])
            print e,
            print "\t%f" % round(mars_rmsep,4),
            print "\t%f" % round(ct_rmsep,4),
            print "\t%f" % round((mars_rmsep-ct_rmsep)*100/mars_rmsep,4)
        print '-----------------------'
        print "Sample\tMars\t\tCalTran\t\t%Gain/lost"
        for i,n in enumerate(names):
            mars_rmsep = rmse(gt_preds[:,i],mars_preds[:,i])
            ct_rmsep = rmse(gt_preds[:,i],ct_preds[:,i])
            print n,
            print "\t%f" % round(mars_rmsep,4),
            print "\t%f" % round(ct_rmsep,4),
            print "\t%f" % round((mars_rmsep-ct_rmsep)*100/mars_rmsep,4)


def run_crossval(spectra,labels,n_folds,transformer):
    train_err = defaultdict(list)
    test_err = defaultdict(list)
    rel_err = defaultdict(list)
    transformed_test = defaultdict(list)
    folds = KFold(spectra[0].shape[0],n_folds)
    for train,test in folds:
        for i in xrange(len(spectra)):
            for j in xrange(len(spectra)):
                #if i==j: # for testing both directions
                if i<=j:
                    continue
                else:
                    A = spectra[i].copy()
                    B = spectra[j].copy()

                    if mean_center:
                        A_shift = A[train].mean(0)
                        B_shift = B[train].mean(0)
                    else:
                        A_shift = 0
                        B_shift = 0
                    A[train] -= A_shift
                    B[train] -= B_shift

                    if scale_std:
                        A_std = A[train].std(0)
                        B_std = B[train].std(0)
                    else:
                        A_std = 1
                        B_std = 1
                    A[train] /= A_std
                    B[train] /= B_std

                    for transform,params in transformers:
                        name = transform.__name__
                        # train
                        proj_to_A,proj_B = transform(A[train],B[train],**params)
                        train_err[name].append(norm(A[train]-proj_B))
                        # test
                        proj_ts_B = A_shift + np.dot((B[test]-B_shift)/B_std,
                                                     proj_to_A*A_std)
                        test_err[name].append(norm(A[test]-proj_ts_B))
                        rel_err[name].append(norm(A[test]-proj_ts_B) /
                                             norm(A[test]-B[test]))
                        transformed_test[name].append(proj_ts_B)
                        if plot_spec:
                            plt.subplot(1,2,1)
                            plt.plot(A[test[0]],c='r',linewidth=.65)
                            plt.plot(B[test[0]],c='b',linewidth=.65)
                            plt.xlim(0,A.shape[1])
                            plt.ylim(ymin=0)
                            plt.title('Alignment error:%.4f' % (
                                norm(A[test[0]]-B[test[0]])))
                            plt.subplot(1,2,2)
                            plt.plot(A[test[0]],c='r',linewidth=.65)
                            plt.plot(proj_ts_B[0],c='b',linewidth=.65)
                            plt.title('Alignment error:%.4f' % (
                                norm(A[test[0]]-proj_ts_B[0])))
                            if experiment is 'caltargets':
                                plt.suptitle('%s - %s - Rel. Error: %.4f\nRed-Earth (target); Blue-Mars (source)' %
                                             (names[test],name,rel_err[name][-1]), size='x-large')
                            else:
                                plt.suptitle('%s Rel. Error: %.4f' % (name,rel_err[name][-1]), size='x-large')
                            plt.xlim(0,A.shape[1])
                            plt.ylim(ymin=0)
                            plt.show()
                        if plot_proj:
                            plt.title(name, size='x-large')
                            plt.imshow(proj_to_A,interpolation=None)
                            plt.colorbar()
                            plt.show()

    for transformer in train_err.keys():
        print "---------------"
        print transformer
        print "train err: %.4f" % np.mean(train_err[transformer])
        print "train std: %.4f" % np.std(train_err[transformer])
        print "test err: %.4f,%.4f" % (np.mean(test_err[transformer]),
                                       np.log10(np.mean(test_err[transformer])))
        print "test std: %.4f" % np.std(test_err[transformer])
        print "rel. err: %.4f" % np.mean(rel_err[transformer])
        print "rel. std: %.4f" % np.std(rel_err[transformer])

    return transformed_test


def make_transform(A,B,ds):
    A_shift = A.mean(0)
    B_shift = B.mean(0)
    shifted_A = A - A_shift
    shifted_B = B - B_shift
    proj_to_A,_ = ds(shifted_A,shifted_B)
    return proj_to_A,A_shift,B_shift


def mars_to_earth_ds():
    ''' 
    To transform mars shots use:
       earth_shots = (mars_shots - mars_shift) * proj_to_earth + earth_shift
    '''
    [earth,mars],_,_ = ct.load_caltargets(normz=ct.norm3)
    proj_to_earth,earth_shift,mars_shift = make_transform(earth,mars,pls_ds)
    return proj_to_earth,earth_shift,mars_shift


if __name__ == '__main__':
    experiment = 'caltargets'
    mean_center=True
    scale_std=False
    plot_proj=False
    plot_spec=False
    profile=False
    verbose=True

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    if experiment is 'noise':
        spectra = load_noise()
        n_folds=5
        transformers = [(direct_standardization,{}),
                        (piecewise_ds,{'win_size':5})]

    elif experiment is 'corn':
        spectra,labels = load_corn()
        n_folds=4
        transformers = [(direct_standardization,{}),
                        (piecewise_ds,{'win_size':3}),
                        (pls_ds,{'n_components':2}),
                        (cca_ds,{'n_components':2}),
                        (new_cca_ds,{'n_components':1}),
                        (ridge_ds,{'rho':1,'beta':0.5}),
                        ]

    elif experiment is 'shootout':
        spectra,labels = load_shootout()
        n_folds=7
        transformers = [(direct_standardization,{}),
                        (piecewise_ds,{'win_size':13}),
                        (pls_ds,{'n_components':3}),
                        (lasso_ds,{})
                        (lowrank_ds,{}),
                        (ridge_ds,{}),
                        ]

    elif experiment is 'caltargets':
        spectra,labels,names = ct.load_caltargets(normz=ct.norm3)
        n_folds=7
        transformers = [
            #(direct_standardization,{}),
            #(piecewise_ds,{'win_size':211,'pls':1}),
            (pls_ds,{'n_components':1}),
            #(cca_ds,{'n_components':1}),
            #(new_cca_ds,{'n_components':1}),
            #(ridge_ds,{}),
            #(lowrank_ds,{}),
        ]


    transformed_test_set = run_crossval(spectra,labels,n_folds,transformers)

    run_full_caltarget_test(spectra,labels,names,transformed_test_set)

    proj_to_earth,earth_shift,mars_shift = mars_to_earth_ds()

    if profile:
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

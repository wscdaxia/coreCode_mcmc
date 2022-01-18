import os
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pylab import mpl
from sklearn import decomposition
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from fig_table.add_batc import handle_m_to_M
from main.tool_3 import mask, find_duplicate
from moregcs.bc03_fit import load_fe_h, load_e_bv, load_model_photometry_interp
from moregcs.vazdekis_fit import load_model_photometry_interp_vaz, add_mags_func_interp
from tools.correction import move
from tools.spline import read_text, read_text_nonsplit
from tools.spline import spline

mpl.rcParams['font.sans-serif'] = ['SimHei']


def fit_error4():  # 用MCMC计算并求误差，vazdekis光谱，vazdekis测光
    # 加载小于1Gyr的星团序号
    lower_index = fits.open('D:/国家天文台/PAPER 1/尽可能多的光谱/vazdekis/lower_1gyr.fits')[0].data
    # 加载chen的波长
    dat_2 = fits.open("D:/国家天文台/PAPER 1/2016-AJ-306个光谱/m31gc/data/GC-9.962540.9696.fits")
    crvalx = dat_2[0].header['CRVAL1']
    cdeltax = dat_2[0].header['CDELT1']
    crpixx = dat_2[0].header['CRPIX1']
    nx = dat_2[0].header['NAXIS1']
    chen_wave = np.arange(-crpixx * cdeltax + crvalx, (nx - crpixx) * cdeltax + crvalx, cdeltax)
    # 加载vazdekis的age_feh
    fe_h = []
    for single in np.arange(-2.32, 0.23, 0.01):
        fe_h += [round(single, 2)] * 49
    arr = fits.open("D:/国家天文台/PAPER 1/全波段拟合/ulyss.1.3.1/ulyss/models/Vaz_Miles.fits")
    age_arr = []
    for single in arr[1].data:
        age_arr.append(np.log10(np.exp(single) * 10 ** 6))
    age_arr = age_arr * 255
    fe_h_age = []
    for i in np.arange(0, len(age_arr)):
        fe_h_age.append([age_arr[i], fe_h[i]])  # [age， fe/h]
    # 加载e(b-v)
    ebv_arr = load_e_bv()
    # 加载vazdekis模板测光数据，转成颜色
    ph_model_arr = load_model_photometry_interp_vaz()
    # 加载GC的测光数据，消光，转成颜色
    judge_arr = []

    fit = fits.open('D:/国家天文台/PAPER 1/搜集的数据/BATC/ascii.fit')[1].data
    sub_judge_arr = []
    batc_mags = []
    lam_batc = [0.4196, 0.4541, 0.4864, 0.5240, 0.5779, 0.6068, 0.6702, 0.7008, 0.7525, 0.8021, 0.8514,
                0.9169, 0.9722]
    for single in fit:
        if min(single[1:14]) > 0:
            e_bv = 0.23
            for single2 in ebv_arr:
                if abs(single[17] - single2[0]) < 0.0004 and abs(single[18] - single2[1]) < 0.0004:
                    e_bv = single2[2]
                    break
            sub_judge_arr.append([single[17], single[18]])
            batc_mag = single[1:14]
            batc_mag = handle_m_to_M(batc_mag, e_bv, lam_batc)
            sub_batc_mags = []
            for i in np.arange(0, 7):
                sub_batc_mags.append(batc_mag[i] - batc_mag[i + 1])
            batc_mags.append(sub_batc_mags)
    judge_arr.append(sub_judge_arr)

    fit = fits.open('D:/国家天文台/PAPER 1/搜集的数据/SDSS-ugriz/new.fit')[1].data
    sub_judge_arr = []
    sdss_mags = []
    lam_sdss = [0.3608, 0.4671, 0.6141, 0.7457, 0.8922]
    for single in fit:
        g = single[3]
        u = single[3] + single[4]
        r = single[3] - single[5]
        i = r - single[6]
        z = i - single[7]
        if min([u, g, r, i, z]) > 0:
            e_bv = 0.23
            for single2 in ebv_arr:
                if abs(single[1] - single2[0]) < 0.0004 and abs(single[2] - single2[1]) < 0.0004:
                    e_bv = single2[2]
                    break
            sub_judge_arr.append([single[1], single[2]])
            sdss_mag = [u, g, r, i, z]
            sdss_mag = handle_m_to_M(sdss_mag, e_bv, lam_sdss)
            sub_sdss_mags = []
            for i in np.arange(1, 2):
                sub_sdss_mags.append(sdss_mag[i] - sdss_mag[i + 1])
            sdss_mags.append(sub_sdss_mags)
    judge_arr.append(sub_judge_arr)
    # 加载模板光谱
    available_wave = np.arange(4000, 7001)
    model_specs = fits.open('D:/国家天文台/PAPER 1/WangSC-APJ/数据/interpolate_model_spec.fits')[0].data
    mixed_model_specs = []
    gc_names = []
    rv = read_text('D:/国家天文台/PAPER 1/尽可能多的光谱/总拟合速度.txt')
    pos_path = read_text_nonsplit('D:/国家天文台/PAPER 1/尽可能多的光谱/总拟合路径.txt')
    arr_path = []
    for single in pos_path:
        arr_path.append(single[0])
    for single in model_specs:
        mixed_model_specs.append(single)
    for i in np.arange(0, len(arr_path)):
        fit = fits.open(arr_path[i])
        flux = []
        try:
            if len(fit[0].data) == 5:
                v = float(rv[i][3]) - 67
                fit[0].data[2] = move(fit[0].data[2], v)
                flux = spline(fit[0].data[2], fit[0].data[0], available_wave)
            if len(fit[0].data) == 2:
                v = float(rv[i][3]) - 67
                new_chen_wave = move(chen_wave, v)
                flux = spline(new_chen_wave, fit[0].data[0], available_wave)
        except Exception:
            if len(fit[1].data) == 5150:
                flux_arr = []
                for single in fit[1].data:
                    flux_arr.append(single[1])
                flux = spline(np.arange(3750., 8900.0), flux_arr, available_wave)
        co = np.polyfit(available_wave, flux, 3)
        p = np.poly1d(co)
        flux = flux / p(available_wave)
        gc_names.append([float(rv[i][1]), float(rv[i][2]), rv[i][0]])
        mixed_model_specs.append(flux)
    spectras = mixed_model_specs
    pca = decomposition.PCA(n_components=10)
    pca.fit(spectras)
    spectra = pca.transform(spectras)
    # 随机森林回归训练
    pca_matrix = []
    pca_matrix_svr = []
    for i in np.arange(0, 12495):
        pca_matrix.append(spectra[i].tolist())
        pca_matrix_svr.append(spectra[i].tolist())
    pca_matrix_svr = np.array(pca_matrix_svr)
    svrs = []
    for i in np.arange(0, 18):
        if i < 10:
            x_train, x_test, y_train, y_test = train_test_split(
                fe_h_age, pca_matrix_svr[:, i], test_size=0.2, random_state=33)
            rbf_svr = ensemble.RandomForestRegressor(n_estimators=4)
            rbf_svr.fit(x_train, y_train)
            print(rbf_svr.score(x_test, y_test))
            # if os.path.exists('D:/国家天文台/PAPER 1/尽可能多的光谱/svrs/model_' + str(i) + '.m'):
            #     os.remove('D:/国家天文台/PAPER 1/尽可能多的光谱/svrs/model_' + str(i) + '.m')
            # joblib.dump(rbf_svr, 'D:/国家天文台/PAPER 1/尽可能多的光谱/svrs/model_' + str(i) + '.m')
            svrs.append(rbf_svr)
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                fe_h_age, ph_model_arr[:, i - 10], test_size=0.2, random_state=33)
            rbf_svr = ensemble.RandomForestRegressor(n_estimators=4)
            rbf_svr.fit(x_train, y_train)
            print(rbf_svr.score(x_test, y_test))
            # if os.path.exists('D:/国家天文台/PAPER 1/尽可能多的光谱/svrs/model_' + str(i) + '.m'):
            #     os.remove('D:/国家天文台/PAPER 1/尽可能多的光谱/svrs/model_' + str(i) + '.m')
            # joblib.dump(rbf_svr, 'D:/国家天文台/PAPER 1/尽可能多的光谱/svrs/model_' + str(i) + '.m')
            svrs.append(rbf_svr)

    # 加载350个GC的测光数据
    status = []  # 标志，记录每个星团含有哪些测光数据
    mixed_spec = []
    for i in np.arange(12495, len(spectra)):
        result = add_mags_func_interp([float(rv[i - 12495][1]), float(rv[i - 12495][2])], batc_mags, sdss_mags,
                                      judge_arr)
        mixed_spec.append(spectra[i].tolist() + result[0])
        status.append(result[1])
    mixed_spec = np.array(mixed_spec)

    nwalkers, ndim = 50, 2
    pos_origin_agefeh = read_text('D:/国家天文台/PAPER 1/尽可能多的光谱/vaz_chi2.txt')
    result = []
    for i in np.arange(87, 174):
        if i in lower_index:
            print(str(len(mixed_spec)) + ':' + str(i))
            age_p0 = float(pos_origin_agefeh[i][3])
            feh_p0 = float(pos_origin_agefeh[i][4])
            if feh_p0 > 0.21:
                feh_p0 = 0.21
            if feh_p0 < -2.31:
                feh_p0 = -2.31
            p0 = np.array(
                [[age_p0, feh_p0] + 1e-4 * np.random.randn(ndim) for j in range(nwalkers)])  # 以传统拟合的age,feh来构造初始p0
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(mixed_spec[i], svrs, status[i]))
            sampler.run_mcmc(p0, 600)
            # pos, _, _ = sampler.run_mcmc(p0, 500)
            # sampler.reset()
            # sampler.run_mcmc(pos, 500)
            samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
            # age_mcmc, feh_mcmc = map(lambda vector: (vector[0], vector[1] - vector[0]), zip(
            #     *np.percentile(samples, [50, 60], axis=0)))
            age_mcmc, feh_mcmc = map(lambda vector: (vector[1], vector[2] - vector[1], vector[1] - vector[0]), zip(
                *np.percentile(samples, [16, 50, 84], axis=0)))
            fig = corner.corner(samples, labels=["$age$", "$feh$"])
            if os.path.exists('D:/国家天文台/PAPER 1/尽可能多的光谱/拟合结果/VAZDEKIS检查图/' + str(i) + '.png'):
                os.remove('D:/国家天文台/PAPER 1/尽可能多的光谱/拟合结果/VAZDEKIS检查图/' + str(i) + '.png')
            fig.savefig('D:/国家天文台/PAPER 1/尽可能多的光谱/拟合结果/VAZDEKIS检查图/' + str(i) + '.png')
            plt.close('all')
            string = rv[i][0] + ' ' + rv[i][1] + ' ' + rv[i][2] + ' ' + str(
                round(age_mcmc[0], 3)) + ' ' + str(round(age_mcmc[1], 3)) + ' ' + str(
                round(feh_mcmc[0], 3)) + ' ' + str(round(feh_mcmc[1], 3)) + '\n'
            result.append(string)
    if os.path.exists('D:/国家天文台/PAPER 1/尽可能多的光谱/642个光谱/除去前面554个剩下的/MCMC拟合_2.txt'):
        os.remove('D:/国家天文台/PAPER 1/尽可能多的光谱/642个光谱/除去前面554个剩下的/MCMC拟合_2.txt')
    f = open('D:/国家天文台/PAPER 1/尽可能多的光谱/642个光谱/除去前面554个剩下的/MCMC拟合_2.txt', 'a')
    f.writelines(result)
    f.close()


def lnprob(theta, spec, svrs, status):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, spec, svrs, status)


def lnprior(theta):
    age, feh = theta
    # if -2.32 <= feh <= 0.22 and 4.0943446 <= age <= 9.78583:
    if -2.32 <= feh <= 0.22 and 7.77 <= age <= 10.25:
        return 0.0
    return -np.inf


def lnlike(theta, spec, svrs, status):
    # total = np.arange(0, 20).tolist()
    # for single in status:
    #     if single == 0:
    #         total += [20, 21, 22, 23, 24, 25, 26]
    #     if single == 1:
    #         total += [27]
    #     if single == 2:
    #         total += [28]
    total = np.arange(0, 10).tolist()
    for single in status:
        if single == 0:
            total += [10, 11, 12, 13, 14, 15, 16]
        if single == 1:
            total += [17]
    age, feh = theta
    spec_simu = []
    for i in total:
        spec_simu.append(svrs[i].predict([[age, feh]])[0])
    spec_simu = np.array(spec_simu)
    return -0.5 * ((np.linalg.norm(spec[0:10] - spec_simu[0:10])) ** 2)
    # return -0.5 * ((np.linalg.norm(spec[0:10] - spec_simu[0:10])) ** 2 + (
    #     np.linalg.norm(spec[10:len(spec)] - spec_simu[10:len(spec)])) ** 2)


fit_error4()

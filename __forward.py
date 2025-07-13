from .__basics import *
from .__utils import *


class FORWARD_GAS_MODEL:
    def __init__(self, param, retrieval=True, canc_metadata=False):
        self.param = copy.deepcopy(param)
        self.process = str(self.param['core_number']) + str(random.randint(0, 100000)) + alphabet() + alphabet() + alphabet() + str(random.randint(0, 100000))
        self.package_dir = param['pkg_dir']
        self.retrieval = retrieval
        self.canc_metadata = canc_metadata
        self.hazes_calc = param['hazes']
        self.c_code_directory = self.package_dir + 'forward_gas_mod/'
        self.matlab_code_directory = self.c_code_directory + 'PlanetModel/'
        try:
            self.working_dir = param['wkg_dir']
        except KeyError:
            self.working_dir = os.getcwd()

    def __atmospheric_structure(self):
        kb = const.k_B.value
        AMU = const.u.value
        RGAS = const.R.value  # Universal Gas Constant, SI
        xH2 = 0.86
        g = self.param['gp']
        meta = self.param['meta']
        opar = self.param['opar']
        Tirr = self.param['Tirr']
        Tint = self.param['Tint']
        KE = self.param['KE']
        cloudfrac = 0.5

        try:
            os.mkdir(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
        except OSError:
            self.process = alphabet() + str(random.randint(0, 100000))
            os.mkdir(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')

        outdir = self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/'

        wave = self.param['wavelength_planet']  # in nanometer

        #    Solar Spectrum
        solar = self.param['solar']

        #    Methane Opacity
        crossCH4 = self.param['crossCH4']

        #    Ammonia Opacity
        crossNH3 = self.param['crossNH3']

        #    Water Opacity
        crossH2O = self.param['crossH2O']

        #    Rayleigh Opacity
        x = wave / 1000.0
        nHe = 1 + (0.01470091 / (423.98 - (x ** (-2.))))
        nH2 = 1.0001393 * np.ones(len(wave))
        DenS = 101325.0 / kb / 273.0 * 1.0E-6
        crossHe = 8.0 * (math.pi ** 3.) * (((nHe ** 2.) - 1) ** 2./3.) / ((wave * 1.0E-7) ** 4.) / DenS / DenS
        crossH2 = 8.0 * (math.pi ** 3.) * (((nH2 ** 2.) - 1) ** 2./3.) / ((wave * 1.0E-7) ** 4.) / DenS / DenS
        crossRay = crossH2 * xH2 + crossHe * (1 - xH2)

        #    Set up pressure grid
        P = self.param['P']  # in Pascal

        #    Convert pressure to column mass g/cm2
        m = P / g * 1.0E-1

        #    Initial Temperature
        T = Tirr * np.ones(len(P))

        watermix = self.param['watermix']
        ammoniamix = self.param['ammoniamix']
        ifwaterc = self.param['cld_pos'][:, 0]
        ifammoniac = self.param['cld_pos'][:, 1]

        #    Atmospheric Composition
        fH2 = self.param['vmr_H2']
        fHe = self.param['vmr_He']
        fCH4 = self.param['vmr_CH4']
        # fH2O = self.param['vmr_H2O']
        # fNH3 = self.param['vmr_NH3']
        # fH2S = self.param['vmr_H2S']
        hazemix = self.param['fhaze']
        MMM = self.param['mean_mol_weight']

        #    Load Mie Calculation Results
        data = np.loadtxt('CrossP/cross_H2OLiquid_M.dat')
        H2OL_r = data[:, 0]  # zero-order radius, in micron
        H2OL_c = data[:, 3]  # cross section per droplet, in cm2
        data = np.loadtxt('CrossP/cross_H2OIce_M.dat')
        H2OI_r = data[:, 0]
        H2OI_c = data[:, 3]
        data = np.loadtxt('CrossP/cross_NH3Ice_M.dat')
        NH3I_r = data[:, 0]
        NH3I_c = data[:, 3]

        miu = np.linspace(0.0001, 0.9999, num=10)
        miuc = np.trapz((miu ** 2.), x=miu) / (1./3.)
        imiu = np.array([1. / miu]).T

        #    Loop to update temperature
        LoopMax = 10
        LoopCri = 1E-4
        LoopVar = 1
        LoopID = 1
        albedo = 0.343  # initial albedo
        tau = np.empty(len(P))
        EE = np.empty(len(P))
        opa = np.empty(len(P))
        Tnew4 = np.empty(len(P))
        cpH2 = np.empty(len(P))
        cpHe = np.empty(len(P))
        cpCH4 = np.empty(len(P))
        cpH2O = np.empty(len(P))
        cloudden = 1.0e-36 * np.ones(len(P))
        cloudopacity = 1.0e-36 * np.ones(len(P))
        particlesize = 1.0e-36 * np.ones(len(P))
        cloudmden = 1.0e-36 * np.ones(len(P))
        cloudmopacity = 1.0e-36 * np.ones(len(P))
        particlemsize = 1.0e-36 * np.ones(len(P))
        redx = np.empty((len(P)-1, len(crossRay)))
        wedx = np.empty((len(P)-1, len(crossRay)))

        Lw1 = 2500800  # latent heat of water evaporation, J/kg at 0C
        Lw2 = 2834100  # latent heat of water sublimation, J/kg at 0C
        LNH3 = 1371200  # latent heat of NH3 vaporization, J/kg at -33.5C

        tg = self.param['t_g']
        pg = self.param['p_g']
        rs = self.param['r_s'] * opar

        while LoopID < LoopMax and LoopVar > LoopCri:
            for i in range(0, len(P)):
                if i == 0:
                    temporaneo = interp2d(tg, np.log10(pg), np.log10(rs), kind='linear')
                    opa[i] = 10. ** temporaneo(min(max(75., T[i]), 4000.), np.log10(min(3.0E+8, max(3.0E+2, P[i]))))
                    tau[i] = m[i] * opa[i] * (5. ** meta)
                else:
                    temporaneo = interp2d(tg, np.log10(pg), np.log10(rs), kind='linear')
                    opa[i] = 10. ** temporaneo(min(max(75., T[i]), 4000.), np.log10(min(3.0E+8, max(3.0E+2, P[i]))))
                    tau[i] = tau[i - 1] + (m[i] - m[i-1]) * opa[i] * (5. ** meta)

            #    Calculate Gamma
            gamma = round(0.13 * np.sqrt(Tirr / 2000.), 4)

            #    Calculate Tnew
            Teq = Tirr * (1./2.) ** (1./2.)
            tck = interp1d(self.param['E2'][:, 0], self.param['E2'][:, 1])
            for i in range(0, len(EE)):
                EE[i] = tck(min(1.0E+5, max(1.0E-3, gamma * tau[i])))
                Tnew4[i] = (0.75 * (Tint ** 4.) * ((2./3.) + tau[i])) + ((1 - albedo) * 0.75 * (Teq ** 4.)) * ((2./3.) + (2./(3. * gamma)) * (1. + ((gamma * tau[i] / 2.) - 1.) * np.exp(-gamma * tau[i])) + ((2. * gamma) / 3.) * (1. - ((tau[i] ** 2.) / 2.)) * EE[i])
            Tnew = Tnew4 ** 0.25

            #    Calculate Molar Heat Capacity
            for i in range(0, len(Tnew)):
                th = min(6000, max(298, Tnew[i]))
                t = th / 1000.0
            #        H2
                if th < 1000.0:
                    A = 33.066178
                    B = -11.363417
                    C = 11.432816
                    D = -2.772874
                    E = -0.158558
                elif th < 2500.0:
                    A = 18.563083
                    B = 12.257357
                    C = -2.859786
                    D = 0.268238
                    E = 1.97799
                else:
                    A = 43.41356
                    B = -4.293079
                    C = 1.272428
                    D = -0.096876
                    E = -20.533862
                cpH2[i] = A + (B * t) + (C * t * t) + (D * t * t * t) + (E / t / t)
            #        He
                A = 20.78603
                B = 4.850638E-10
                C = -1.582916E-10
                D = 1.525102E-11
                E = 3.196347E-11
                cpHe[i] = A + (B * t) + (C * t * t) + (D * t * t * t) + (E / t / t)
            #        CH4
                if th < 1300.0:
                    A = -0.703029
                    B = 108.4773
                    C = -42.52157
                    D = 5.862788
                    E = 0.678565
                else:
                    A = 85.81217
                    B = 11.26467
                    C = -2.114146
                    D = 0.138190
                    E = -26.42221
                cpCH4[i] = A + (B * t) + (C * t * t) + (D * t * t * t) + (E / t / t)
            #        H2O
                th = min(6000, max(500, Tnew[i]))
                t = th / 1000.0
                if th < 1700.0:
                    A = 30.09200
                    B = 6.832514
                    C = 6.793435
                    D = -2.534480
                    E = 0.082139
                else:
                    A = 41.96426
                    B = 8.622053
                    C = -1.499780
                    D = 0.098119
                    E = -11.15764
                cpH2O[i] = A + (B * t) + (C * t * t) + (D * t * t * t) + (E / t / t)
            # cp = (cpH2 * fH2) + (cpHe * fHe) + (cpH2O * fH2O) + (cpCH4 * fCH4)
            cp = (cpH2 * fH2 + cpHe * fHe) / (fH2 + fHe)

            #    Calculate lapse rate
            cpm = cp / (MMM * 0.001)  # convert to J/kg/K
            lapse = kb / MMM / AMU / cpm  # dry adiabatic lapse rate for dlnT/dlnP dimensionless

            #    Process the temperature profile
            for i in range(0, len(P)-1):
                dlnTdlnP = (np.log(Tnew[i + 1]) - np.log(Tnew[i])) / (np.log(P[i + 1]) - np.log(P[i]))
                if dlnTdlnP > lapse[i]:
                    Tnew[i + 1] = np.exp(np.log(Tnew[i]) + (np.log(P[i + 1]) - np.log(P[i])) * lapse[i])

            #    Check atmosphere stability and cloud formation
            for i in range(len(P) - 2, -1, -1):
                if ifwaterc[i] != 0.0 and ifammoniac[i] == 0.0:  # condensation of water only
                    deltaP = P[i] * abs(watermix[i] - watermix[i + 1])
                    cloudden[i] = max(abs(watermix[i] - watermix[i + 1]) * 0.018 * P[i] / RGAS / Tnew[i], 1e-16)  # kg/m^3, g/L
            #       calculate cloud particle size
                    r0, r1, r2, VP = particlesizef(g, Tnew[i], P[i], MMM, 18.0, KE, deltaP)
                    particlesize[i] = r2
            #       calculate moist lapse rate and cloud opacity profile
                    if Tnew[i] < 273.16:  # ice
                        lapsem = lapse[i] * (1 + Lw2 * 0.018 * watermix[i] / RGAS / Tnew[i]) / (1 + (Lw2 ** 2.) * (0.018 ** 2.) * watermix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i])
                        tck = interp1d(np.log10(H2OI_r), np.log10(H2OI_c))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        cloudopacity[i] = cloudden[i] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
                    else:  # liquid
                        lapsem = lapse[i] * (1 + Lw1 * 0.018 * watermix[i] / RGAS / Tnew[i]) / (1 + (Lw1 ** 2.) * (0.018 ** 2.) * watermix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i])
                        tck = interp1d(np.log10(H2OL_r), np.log10(H2OL_c))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        cloudopacity[i] = cloudden[i] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
            #       see if convect
                    dlnTdlnP = (np.log(Tnew[i-1]) - np.log(Tnew[i])) / (np.log(P[i - 1]) - np.log(P[i]))
                    if dlnTdlnP > lapsem:
                        Tnew[i-1] = np.exp(np.log(Tnew[i]) + ((np.log(P[i - 1]) - np.log(P[i])) * lapsem))

                elif ifammoniac[i] != 0.0 and ifwaterc[i] == 0.0:  # condensation of ammonia only
                    deltaP = P[i] * abs(ammoniamix[i] - ammoniamix[i + 1])
                    cloudmden[i] = max(abs(ammoniamix[i] - ammoniamix[i + 1]) * 0.017 * P[i] / RGAS / Tnew[i], 1e-16)  # kg/m^3, g/L
            #       calculate cloud particle size
                    r0, r1, r2, VP = particlesizef(g, Tnew[i], P[i], MMM, 17.0, KE, deltaP)
                    particlemsize[i] = r2
            #       calculate moist lapse rate and cloud opacity profile
                    lapsem = lapse[i] * (1 + LNH3 * 0.017 * ammoniamix[i] / RGAS / Tnew[i]) / (1 + (LNH3 ** 2.) * (0.017 ** 2.) * ammoniamix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i])
                    tck = interp1d(np.log10(NH3I_r), np.log10(NH3I_c))
                    temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                    cloudmopacity[i] = cloudmden[i] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
            #       see if convect
                    dlnTdlnP = (np.log(Tnew[i - 1]) - np.log(Tnew[i])) / (np.log(P[i - 1]) - np.log(P[i]))
                    if dlnTdlnP > lapsem:
                        Tnew[i - 1] = np.exp(np.log(Tnew[i]) + ((np.log(P[i-1]) - np.log(P[i])) * lapsem))

                elif ifammoniac[i] != 0.0 and ifwaterc[i] != 0.0:  # both water and ammonia condense
                    deltaP1 = P[i] * abs(watermix[i] - watermix[i + 1])
                    cloudden[i] = max(abs(watermix[i] - watermix[i + 1]) * 0.018 * P[i] / RGAS / Tnew[i], 1e-16)  # kg/m^3, g/L
                    deltaP2 = P[i] * abs(ammoniamix[i] - ammoniamix[i + 1])
                    cloudmden[i] = max(abs(ammoniamix[i] - ammoniamix[i + 1]) * 0.017 * P[i] / RGAS / Tnew[i], 1e-16)  # kg/m^3, g/L
            #       calculate cloud particle size and cloud opacity profile
                    r0, r1, r2, VP = particlesizef(g, Tnew[i], P[i], MMM, 18.0, KE, deltaP1)
                    particlesize[i] = r2
                    if Tnew[i] < 273.16:  # ice
                        tck = interp1d(np.log10(H2OI_r), np.log10(H2OI_c))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        cloudopacity[i] = cloudden[i] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
                    else:  # liquid
                        tck = interp1d(np.log10(H2OL_r), np.log10(H2OL_c))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        cloudopacity[i] = cloudden[i] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
            #       calculate cloud particle size
                    r0, r1, r2, VP = particlesizef(g, Tnew[i], P[i], MMM, 17.0, KE, deltaP2)
                    particlemsize[i] = r2
                    tck = interp1d(np.log10(NH3I_r), np.log10(NH3I_c))
                    temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                    cloudmopacity[i] = cloudmden[i] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
            #       calculate moist lapse rate
                    if Tnew[i] < 273.16:  # ice
                        lapsem = lapse[i] * ((1 + Lw2 * 0.018 * watermix[i] / RGAS / Tnew[i]) + (LNH3 * 0.017 * ammoniamix[i] / RGAS / Tnew[i])) / (1 + ((Lw2 ** 2.) * (0.018 ** 2.) * watermix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i]) + ((LNH3 ** 2.) * (0.017 ** 2.) * ammoniamix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i]))
                    else:  # liquid
                        lapsem = lapse[i] * ((1 + Lw1 * 0.018 * watermix[i] / RGAS / Tnew[i]) + (LNH3 * 0.017 * ammoniamix[i] / RGAS / Tnew[i])) / (1 + ((Lw1 ** 2.) * (0.018 ** 2.) * watermix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i]) + ((LNH3 ** 2.) * (0.017 ** 2.) * ammoniamix[i] / MMM / 0.001 / RGAS / (Tnew[i] ** 2.) / cpm[i]))
            #       see if convect
                    dlnTdlnP = (np.log(Tnew[i - 1]) - np.log(Tnew[i])) / (np.log(P[i - 1]) - np.log(P[i]))
                    if dlnTdlnP > lapsem:
                        Tnew[i - 1] = np.exp(np.log(Tnew[i]) + ((np.log(P[i - 1]) - np.log(P[i])) * lapsem))

                else:  # no condensation
                    dlnTdlnP = (np.log(Tnew[i - 1]) - np.log(Tnew[i])) / (np.log(P[i-1]) - np.log(P[i]))
                    if dlnTdlnP > lapse[i]:
                        Tnew[i - 1] = np.exp(np.log(Tnew[i]) + ((np.log(P[i - 1]) - np.log(P[i])) * lapse[i]))

            #    Update albedo
            clouddepth = 0
            wclouddepth = 0
            rayleighdepth = 0
            albedod = np.ones((len(miu), len(wave)))
            albedod0 = np.ones((len(miu), len(wave)))
            cloudmass = 0
            cloudmmass = 0

            for i in range(0, len(P)-1):
                sh = kb * T[i] / MMM / AMU / g * 100  # cm
                clouddepth = clouddepth + (cloudopacity[i] + cloudmopacity[i]) * np.log(P[i + 1] / P[i]) * sh  # cloud
                wclouddepth = wclouddepth + cloudopacity[i] * np.log(P[i + 1] / P[i]) * sh  # water cloud
                cloudmass = cloudmass + cloudden[i] * np.log(P[i + 1] / P[i]) * sh / 100
                cloudmmass = cloudmmass + cloudmden[i] * np.log(P[i + 1] / P[i]) * sh / 100
                rayleighdepth = rayleighdepth + crossRay * (P[i + 1] - P[i]) / g * 1.0E-4 / MMM / AMU
                refdepth = clouddepth + rayleighdepth
                wrefdepth = wclouddepth + rayleighdepth
                ridx = (np.array(np.dot(imiu, np.array([refdepth]))) < 1).astype(int)
                wridx = (np.array(np.dot(imiu, np.array([wrefdepth]))) < 1).astype(int)
                gasopa = (crossCH4 * fCH4 + crossH2O * watermix[i] + crossNH3 * ammoniamix[i]) * (P[i + 1] - P[i]) / g * 1.0E-4 / MMM / AMU
                albedod = albedod * np.exp(-2. * ridx * (np.dot(imiu, np.array([gasopa]))))
                albedod0 = albedod0 * np.exp(-2. * wridx * (np.dot(imiu, np.array([gasopa]))))
                redx[i, :] = 1 - ridx[-1, :]
                wedx[i, :] = 1 - wridx[-1, :]

            albedoD, albedoD0 = np.ones(len(wave)), np.ones(len(wave))
            tempa = albedod * (np.dot(np.array([miu]).T, np.array([np.ones(len(wave))])) ** 2.)
            tempb = albedod0 * (np.dot(np.array([miu]).T, np.array([np.ones(len(wave))])) ** 2.)
            for j in range(0, len(albedoD)):
                albedoD[j] = 2. * np.trapz(tempa[:, j], x=miu) / miuc
                albedoD0[j] = 2. * np.trapz(tempb[:, j], x=miu) / miuc

            id1, id2 = np.empty(len(redx[0, :])), np.empty(len(redx[0, :]))
            for i in range(0, len(redx[0, :])):
                id1[i] = np.max(redx[:, i])
                id2[i] = np.argmax(redx[:, i])
            Pref = P[id2.astype(int)]
            for i in range(0, len(wedx[0, :])):
                id1[i] = np.max(wedx[:, i])
                id2[i] = np.argmax(wedx[:, i])
            wPref = P[id2.astype(int)]

            albedoD = (albedoD * cloudfrac + albedoD0 * (1. - cloudfrac)) / (2. / 3.) * 0.55  # 17.5% reduction due to Raman/Haze
            albedo = np.dot(albedoD, np.array([solar]).T) / np.sum(solar) * 0.8  # the phase integral = 0.67+-0.06 from 4 SS giant planets % assumed to be 1.25 Marley et al. (1999)

            #    Update LoopVar
            LoopVar = max(abs(Tnew - T) / Tnew)
            T = Tnew + 0.0
            #    Iterate
            LoopID += 1

        self.param['albedo'] = albedo
        if not self.retrieval:
            np.savetxt(outdir + 'watermix.dat', watermix)
            np.savetxt(outdir + 'ammoniamix.dat', ammoniamix)

            np.savetxt(outdir + 'particlesize.dat', particlesize)
            np.savetxt(outdir + 'particlemsize.dat', particlemsize)

            np.savetxt(outdir + 'cloudden.dat', cloudden)
            np.savetxt(outdir + 'cloudmden.dat', cloudmden)

            np.savetxt(outdir + 'P.dat', P)
            np.savetxt(outdir + 'T.dat', T)
            np.savetxt(outdir + 'tau.dat', tau)

        #    Calculate the height
        P = P[::-1]
        T = T[::-1]
        watermix = watermix[::-1]
        ammoniamix = ammoniamix[::-1]
        cloudden = cloudden[::-1]
        cloudmden = cloudmden[::-1]
        particlesize = particlesize[::-1]
        particlemsize = particlemsize[::-1]
        Z = np.zeros(len(P))
        for j in range(0, len(P)-1):
            H = kb * (T[j] + T[j + 1]) / 2. / g / MMM / AMU / 1000.  # km
            Z[j + 1] = Z[j] + H * np.log(P[j] / P[j + 1])

        #    Re-sample in equal height spacing
        nn = 181
        zz = np.linspace(Z[0], Z[-1], num=nn)
        z0 = zz[0: nn-1]
        z1 = zz[1: nn]
        zl = z0 * 0.5 + z1 * 0.5

        tck = interp1d(Z, T)
        tl = tck(zl)

        tck = interp1d(Z, np.log(P))
        pl = np.exp(tck(zl))

        nden = pl / kb / tl * 1.0E-6  # molecule cm-3
        nH2 = fH2 * nden
        # nHe = fHe * nden
        nCH4 = fCH4 * nden

        tck = interp1d(Z, np.log(watermix))
        nH2O = np.exp(tck(zl)) * nden
        tck = interp1d(Z, np.log(ammoniamix))
        nNH3 = np.exp(tck(zl)) * nden
        tck = interp1d(Z, np.log(cloudden))
        cloudden = np.exp(tck(zl))
        tck = interp1d(Z, np.log(cloudmden))
        cloudmden = np.exp(tck(zl))
        tck = interp1d(Z, np.log(particlesize))
        particlesize = np.exp(tck(zl))
        tck = interp1d(Z, np.log(particlemsize))
        particlemsize = np.exp(tck(zl))

        #    Generate ConcentrationSTD.dat file
        NSP = 111
        with open(outdir + 'ConcentrationSTD.dat', 'w') as file:
            file.write('z\t\tz0\t\tz1\t\tT\t\tP')
            for i in range(1, NSP + 1):
                file.write('\t\t' + str(i))
            file.write('\n')
            file.write('km\t\tkm\t\tkm\t\tK\t\tPa\n')
            for j in range(0, len(zl)):
                file.write(str(zl[j]) + '\t\t' + str(z0[j]) + '\t\t' + str(z1[j]) + '\t\t' + str(tl[j]) + '\t\t' + str(pl[j]))
                for i in range(1, NSP + 1):
                    if i == 7:
                        file.write('\t\t' + str(nH2O[j]))
                    elif i == 21:
                        file.write('\t\t' + str(nCH4[j]))
                    elif i == 9:
                        file.write('\t\t' + str(nNH3[j]))
                    elif i == 53:
                        file.write('\t\t' + str(nH2[j]))
        #        elseif i==78\n',
        #            fprintf(f,'%.6e\\t',nH2Ol(j));\n",
        #        elseif i==111\n',
        #            fprintf(f,'%.6e\\t',nNH3l(j));\n",
                    else:
                        file.write('\t\t' + str(0.0))
                file.write('\n')

        #    cloud output
        H2OL_r = self.param['H2OL_r']  # zero-order radius, in micron
        H2OL_c = self.param['H2OL_c']  # cross section per droplet, in cm2
        H2OL_a = self.param['H2OL_a']  # albedo
        H2OL_g = self.param['H2OL_g']  # geometric albedo

        H2OI_r = self.param['H2OI_r']
        H2OI_c = self.param['H2OI_c']
        H2OI_a = self.param['H2OI_a']
        H2OI_g = self.param['H2OI_g']

        NH3I_r = self.param['NH3I_r']
        NH3I_c = self.param['NH3I_c']
        NH3I_a = self.param['NH3I_a']
        NH3I_g = self.param['NH3I_g']

        croa = np.zeros((len(zl), 5))
        alba = np.ones((len(zl), 5))
        geoa = np.zeros((len(zl), 5))
        crow = np.zeros((len(zl), 5))
        albw = np.ones((len(zl), 5))
        geow = np.zeros((len(zl), 5))

        #    opacity
        sig = 2
        for j in range(0, len(zl)):
            r2 = particlemsize[j]
            if r2 < 1e-1:
                pass
            else:
                r0 = r2 * np.exp(-np.log(sig) ** 2.)  # micron
                VP = 4. * math.pi / 3. * ((r2 * 1.0E-6 * np.exp(0.5 * (np.log(sig) ** 2.))) ** 3.) * 1.0E+6 * 0.87  # g
                for indi in range(0, 5):
                    tck = interp1d(np.log10(NH3I_r), np.log10(NH3I_c[:, indi]))
                    temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                    croa[j, indi] = cloudmden[j] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
                    tck = interp1d(np.log10(NH3I_r), NH3I_a[:, indi])
                    alba[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                    tck = interp1d(np.log10(NH3I_r), NH3I_g[:, indi])
                    geoa[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
            r2 = particlesize[j]
            if r2 < 1e-1:
                pass
            else:
                r0 = r2 * np.exp(-np.log(sig) ** 2.)
                if tl[j] < 273.16:  # ice
                    VP = 4. * math.pi / 3. * ((r2 * 1.0E-6 * np.exp(0.5 * np.log(sig) ** 2)) ** 3.) * 1.0E+6 * 0.92  # g
                    for indi in range(0, 5):
                        tck = interp1d(np.log10(H2OI_r), np.log10(H2OI_c[:, indi]))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        crow[j, indi] = cloudden[j] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
                        tck = interp1d(np.log10(H2OI_r), H2OI_a[:, indi])
                        albw[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                        tck = interp1d(np.log10(H2OI_r), H2OI_g[:, indi])
                        geow[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                else:  # liquid
                    VP = 4. * math.pi / 3. * ((r2 * 1.0E-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0E+6 * 1.0  # g
                    for indi in range(0, 5):
                        tck = interp1d(np.log10(H2OL_r), np.log10(H2OL_c[:, indi]))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        crow[j, indi] = cloudden[j] / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
                        tck = interp1d(np.log10(H2OL_r), H2OL_a[:, indi])
                        albw[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                        tck = interp1d(np.log10(H2OL_r), H2OL_g[:, indi])
                        geow[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))

        tck = interp1d(wave, Pref)
        Prefref = tck(800)
        tck = interp1d(wave, wPref)
        wPrefref = tck(800)
        hazetau = np.zeros(5)

        #    add haze opacity
        if self.hazes_calc:
            hazeopa = np.ones(5)
            hazealb = np.ones(5)
            hazegeo = np.ones(5)

            for j in range(0, len(zl)):
                if Prefref * np.exp(-2.) <= pl[j] <= Prefref:
                    hazeden = hazemix * 0.017 * pl[j] / RGAS / tl[j]  # kg/m^3, g/L
                    r0, r1, r2, VP = particlesizef(g, tl[j], pl[j], MMM, 17.0, KE, 0.01 * hazemix * pl[j])
                    for indi in range(0, 5):
                        tck = interp1d(np.log10(NH3I_r), np.log10(NH3I_c[:, indi]))
                        temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                        hazeopa[indi] = hazeden / VP * 1.0E-3 * (10. ** temporaneo)  # cm-1
                        tck = interp1d(np.log10(NH3I_r), NH3I_a[:, indi])
                        hazealb[indi] = tck(np.log10(max(0.01, min(r0, 100))))
                        tck = interp1d(np.log10(NH3I_r), NH3I_g[:, indi])
                        hazegeo[indi] = tck(np.log10(max(0.01, min(r0, 100))))
                        hazetau[indi] = hazetau[indi] + hazeopa[indi] * (z1[j] - z0[j]) * 1E+5

                        if hazemix > 0:
                            alba[j, indi] = (alba[j, indi] * croa[j, indi] + hazealb[indi] * hazeopa[indi]) / (croa[j, indi] + hazeopa[indi])
                            geoa[j, indi] = (geoa[j, indi] * croa[j, indi] + hazegeo[indi] * hazeopa[indi]) / (croa[j, indi] + hazeopa[indi])
                            croa[j, indi] = croa[j, indi] + hazeopa[indi]

        tck = interp1d([450, 550, 675, 825, 975], hazetau)
        hopticaldepth = tck(800)

        with open(outdir + 'Cloudtop.dat', 'w') as file:
            file.write('Cloud top Pressure is at ' + str(Prefref) + '\n')
            file.write('Water cloud top Pressure is at ' + str(wPrefref) + '\n')
            file.write('Photochemical haze has optical depth of ' + str(hopticaldepth) + '\n')

        np.savetxt(outdir + 'cross_H2O.dat', crow)
        np.savetxt(outdir + 'cross_NH3.dat', croa)

        np.savetxt(outdir + 'albedo_H2O.dat', albw)
        np.savetxt(outdir + 'albedo_NH3.dat', alba)

        np.savetxt(outdir + 'geo_H2O.dat', geow)
        np.savetxt(outdir + 'geo_NH3.dat', geoa)

    def __run_structure(self):
        os.chdir(self.matlab_code_directory)
        self.__atmospheric_structure()
        os.chdir(self.working_dir)

    def __par_c_file(self):
        c_par_file = ['#ifndef _PLANET_H_\n',
                      '#define _PLANET_H_ \n',
                      #
                      # Planet Physical Properties
                      '#define MASS_PLANET          ' + str(self.param['Mp'] * const.M_jup.value) + '\n',  # kg
                      '#define RADIUS_PLANET        ' + str(self.param['Rp'] * const.R_jup.value) + '\n',  # m
                      #
                      # Planet Orbital Properties
                      '#define ORBIT                ' + str(self.param['equivalent_a']) + '\n',  # AU
                      '#define STAR_SPEC            "Data/solar0.txt"\n',
                      '#define TIDELOCK             0\n',  # If the planet is tidally locked
                      '#define FaintSun             1.0\n',  # Faint early Sun factor
                      '#define STAR_TEMP            394.109\n',  # str(self.param['Tirr'] / (self.param['major-a'] ** 0.5))  irradiation Temperature at 1 AU
                      '#define THETAREF             1.0471\n',  # Slant Path Angle in radian
                      '#define PAB                  ' + str(self.param['albedo']) + '\n',  # Planet Bond Albedo
                      '#define FADV                 0.25\n',  # Advection factor: 0.25=uniformly distributed, 0.6667=no Advection
                      '#define PSURFAB              0\n',  # Planet Surface Albedo
                      '#define PSURFEM              1.0\n',  # Planet Surface Emissivity
                      '#define DELADJUST            1\n',  # Whether use the delta adjustment in the 2-stream diffuse radiation
                      '#define TAUTHRESHOLD         0.1\n',  # Optical Depth Threshold for multi-layer diffuse radiation
                      '#define TAUMAX               1000.0\n',  # Maximum optical Depth in the diffuse radiation
                      '#define TAUMAX1              1000.0\n',  # Maximum optical Depth in the diffuse radiation
                      '#define TAUMAX2              1000.0\n',
                      '#define IFDIFFUSE            1\n',  # Set to 1 if want to include diffuse solar radiation into the photolysis rate
                      #
                      '#define IFUVMULT             0\n',  # Whether do the UV Multiplying
                      '#define FUVMULT              1.0E+3\n',  # Multiplying factor for FUV radiation <200 nm
                      '#define MUVMULT              1.0E+2\n',  # Multiplying factor for MUV radiation 200 - 300 nm
                      '#define NUVMULT              1.0E+1\n',  # Multiplying factor for NUV radiation 300 - 400 nm
                      #
                      # Planet Temperature-Pressure Preofile
                      '#define TPMODE               1\n',  # 1: import data from a ZTP list
                      #                                      0: calculate TP profile from the parametrized formula
                      '#define TPLIST               "Data/TPStdJupiter.dat"\n',
                      '#define PTOP                 1.0E-8\n',  # Pressure at the top of atmosphere in bar
                      '#define TTOP                 480.0\n',  # Temperature at the top of atmosphere
                      '#define TSTR                 550.0\n',  # Temperature at the top of stratosphere
                      '#define TINV                 0\n',  # set to 1 if there is a temperature inversion
                      '#define PSTR                 1.0E-1\n',  # Pressure at the top of stratosphere
                      '#define PMIDDLE              0\n',  # Pressure at the bottom of stratosphere
                      '#define TMIDDLE              0\n',  # Temperature at the bottom of stratosphere
                      '#define PBOTTOM              1.0E+0\n',  # Pressure at the bottom of stratosphere
                      '#define TBOTTOM              1050.0\n',  # Temperature at the bottom of stratosphere
                      '#define PPOFFSET             0.0\n',  # Pressure offset in log [Pa]
                      #
                      # Calculation Grids
                      '#define zbin                 180\n',  # How many altitude bin?
                      # #define zmax 1631.0 Maximum altitude in km
                      # #define zmin 0.0 Maximum altitude in km
                      '#define WaveBin              9999\n',  # How many wavelength bin?, was 9999
                      '#define WaveMin              1.0\n',  # Minimum Wavelength in nm
                      '#define WaveMax              10000.0\n',  # Maximum Wavelength in nm, was 10000
                      '#define WaveMax1             1000.0\n',  # Maximum Wavelength in nm for the Calculation of UV-visible radiation and photolysis rates
                      '#define TDEPMAX              300.0\n',  # Maximum Temperature-dependence Validity for UV Cross sections
                      '#define TDEPMIN              200.0\n',  # Minimum Temperature-dependence Validity for UV Cross sections
                      #
                      # The criteria of convergence
                      '#define Tol1                 1.0E+10\n',
                      '#define Tol2                 1.0E-16\n',
                      #
                      # Mode of iteration
                      '#define TSINI                1.0E-18\n',  # Initial Trial Timestep, generally 1.0E-8
                      '#define FINE1                1\n',  # Set to one for fine iteration: Set to 2 to disregard the bottom boundary layers
                      '#define FINE2                1\n',  # Set to one for fine iteration: Set to 2 to disregard the fastest varying point
                      '#define TMAX                 1.0E+12\n',  # Maximum of time step
                      '#define TMIN                 1.0E-25\n',  # Minimum of time step
                      '#define TSPEED               1.0E+12\n',  # Speed up factor
                      '#define NMAX                 1E+4\n',  # Maximum iteration cycles
                      '#define NMAXT                1.0E+13\n',  # Maximum iteration cumulative time in seconds
                      '#define MINNUM               1.0E-0\n',  # Minimum number density in denominator
                      #
                      # Molecular Species
                      '#define NSP                  111\n',  # Number of species in the standard list
                      '#define SPECIES_LIST         "Data/species_Earth_Full.dat"\n',
                      '#define AIRM                 ' + str(self.param['mean_mol_weight']) + '\n',  # Initial mean molecular mass of atmosphere, in atomic mass unit
                      '#define AIRVIS               1.0E-5\n',  # Dynamic viscosity in SI
                      '#define RefIdxType           6\n',  # Type of Refractive Index: 0=Air, 1=CO2, 2=He, 3=N2, 4=NH3, 5=CH4, 6=H2, 7=O2
                      #
                      # Aerosol Species
                      '#define AERSIZE              1.0E-7\n',  # diameter in m
                      '#define AERDEN               1.84E+3\n',  # density in SI
                      '#define NCONDEN              1\n',  # Calculate the condensation every NCONDEN iterations
                      '#define IFGREYAER            0\n',  # Contribute to the grey atmosphere Temperature? 0=no, 1=yes
                      '#define SATURATIONREDUCTION  1.0\n',  # Ad hoc reduction factor for saturation pressure of water
                      '#define AERRADFILE1          "Data/H2SO4AER_CrossM_01.dat"\n',  # radiative properties of H2SO4
                      '#define AERRADFILE2          "Data/S8AER_CrossM_01.dat"\n',  # radiative properties of S8
                      #
                      # Initial Concentration Setting
                      '#define IMODE                4\n',  # 1: Import from SPECIES_LIST
                      #                                    # 0: Calculate initial concentrations from chemical equilibrium sub-routines (not rad)
                      #                                    # 3: Calculate initial concentrations from simplied chemical equilibrium formula (not rad)
                      #                                    # 2: Import from results of previous calculations
                      #                                    # 4: Import from results of previous calculations in the standard form (TP import only for rad)
                      '#define NATOMS               23\n',  # Number of atoms for chemical equil
                      '#define NMOLECULES           172\n',  # Number of molecules for chemical equil
                      '#define MOL_DATA_FILE        "Data/molecules_all.dat"\n',  # Data file for chemical equilibrium calculation
                      '#define ATOM_ABUN_FILE       "Data/atom_H2O_CH4.dat"\n',  # Data file for chemical equilibrium calculation
                      '#define IMPORTFILEX          "Result/Aux/Conx.dat"\n',  # File of concentrations X to be imported
                      '#define IMPORTFILEF          "Result/Aux/Conf.dat"\n',  # File of concentrations F to be imported
                      '#define IFIMPORTH2O          0\n',  # When H2O is set to constant, 1=import mixing ratios
                      '#define IFIMPORTCO2          0\n',  # When CO is set to constant, 1=import mixing ratios
                      #
                      # Reaction Zones
                      '#define REACTION_LIST        "Data/zone_Earth_Full.dat"\n',
                      '#define NKin                 645\n',  # Number of Regular Chemical Reaction in the standard list
                      '#define NKinM                90\n',  # Number of Thermolecular Reaction in the standard list
                      '#define NKinT                93\n',  # Number of Thermal Dissociation Reaction in the standard list
                      '#define NPho                 71\n',  # Number of Photochemical Reaction in the standard list
                      '#define THREEBODY            1.0\n',  # Enhancement of THREEBODY Reaction when CO2 dominant
                      #
                      # Parametization of Eddy Diffusion Coefficient
                      '#define EDDYPARA             1\n',  # =1 from Parametization, =2 from imported list
                      '#define KET                  1.0E+6\n',  # unit cm2 s-1
                      '#define KEH                  1.0E+6\n',
                      '#define ZT                   200.0\n',  # unit km
                      '#define Tback                1E+4\n',
                      '#define KET1                 1.0E+6\n',
                      '#define KEH1                 1.0E+8\n',
                      '#define EDDYIMPORT           "Data/EddyH2.dat"\n',
                      '#define MDIFF_H_1            4.87\n',
                      '#define MDIFF_H_2            0.698\n',
                      '#define MDIFF_H2_1           2.80\n',
                      '#define MDIFF_H2_2           0.740\n',
                      '#define MDIFF_H2_F           1.0\n',
                      #
                      # Parameters of rainout rates
                      '#define RainF                0.0\n',  # Rainout factor, 0 for no rainout, 1 for earthlike normal rainout, <1 for reduced rainout
                      '#define CloudDen             1.0\n',  # Cloud density in the unit of g m-3
                      #
                      # Output Options
                      '#define OUT_DIR              "Result/Retrieval_' + str(self.process) + '/"\n',
                      '#define TINTSET              ' + str(self.param['Tint']) + '\n',  # Internal Heat Temperature
                      '#define OUT_STD              "Result/Jupiter_1/ConcentrationSTD.dat"\n',
                      '#define OUT_FILE1            "Result/GJ1214_Figure/Conx.dat"\n',
                      '#define OUT_FILE2            "Result/GJ1214_Figure/Conf.dat"\n',
                      '#define NPRINT               1E+2\n',  # Printout results and histories every NPRINT iterations
                      '#define HISTORYPRINT         0\n',  # print out time series of chemical composition if set to 1
                      #
                      # Input choices for the infrared opacities
                      # Must be set to the same as the opacity code
                      #
                      '#define CROSSHEADING         "Cross3/H2_FullT_LowRes/"\n',
                      #
                      '#define NTEMP                20\n',  # Number of temperature points in grid
                      '#define TLOW                 100.0\n',  # Temperature range in K
                      '#define THIGH                2000.0\n',
                      #
                      '#define NPRESSURE            10\n',  # Number of pressure points in grid
                      '#define PLOW                 1.0e-01\n',  # Pressure range in Pa
                      '#define PHIGH                1.0e+08\n',
                      #
                      '#define NLAMBDA              16000\n',  # Number of wavelength points in grid
                      '#define LAMBDALOW            1.0e-07\n',  # Wavelength range in m -> 0.1 micron
                      '#define LAMBDAHIGH           2.0e-04\n',  # in m
                      '#define LAMBDATYPE           1\n',  # LAMBDATYPE=1 -> constant resolution
                      #                                    # LAMBDATYPE=2 -> constant wave step
                      #
                      # IR emission spectra output options
                      '#define IRLamMin             1.0\n',  # Minimum wavelength in the IR emission output, in microns
                      '#define IRLamMax             100.0\n',  # Maximum wavelength in the IR emission output, in microns, was 100
                      '#define IRLamBin             9999\n',  # Number of wavelength bin in the IR emission spectra, was 9999
                      '#define Var1STD              7\n',
                      '#define Var2STD              20\n',
                      '#define Var3STD              21\n',
                      '#define Var4STD              52\n',
                      '#define Var1RATIO            0.0\n',
                      '#define Var2RATIO            0.0\n',
                      '#define Var3RATIO            0.0\n',
                      '#define Var4RATIO            0.0\n',
                      #
                      #  Stellar Light Reflection output options
                      '#define UVRFILE              "Result/Jupiter_1/Reflection"\n',  # Output spectrum file name
                      '#define UVRFILEVar1          "Result/Jupiter_1/ReflectionVar1.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar2          "Result/Jupiter_1/ReflectionVar2.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar3          "Result/Jupiter_1/ReflectionVar3.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar4          "Result/Jupiter_1/ReflectionVar4.dat"\n',  # Output spectrum file name
                      '#define UVROPTFILE           "Result/Jupiter_1/UVROpt.dat"\n',  # Output spectrum file name
                      '#define AGFILE               "Result/Jupiter_1/GeometricA.dat"\n',  # Output spectrum file name
                      #
                      # Stellar Light Transmission output options
                      '#define UVTFILE              "Result/Jupiter_1/Transmission.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar1          "Result/Jupiter_1/TransmissionVar1.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar2          "Result/Jupiter_1/TransmissionVar2.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar3          "Result/Jupiter_1/TransmissionVar3.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar4          "Result/Jupiter_1/TransmissionVar4.dat"\n',  # Output spectrum file name
                      '#define UVTOPTFILE           "Result/Jupiter_1/UVTOpt.dat"\n',  # Output spectrum file name
                      #
                      # Thermal Emission output options
                      '#define IRFILE               "Result/Jupiter_1/Emission.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar1           "Result/Jupiter_1/EmissionVar1.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar2           "Result/Jupiter_1/EmissionVar2.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar3           "Result/Jupiter_1/EmissionVar3.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar4           "Result/Jupiter_1/EmissionVar4.dat"\n',  # Output spectrum file name
                      '#define IRCLOUDFILE          "Result/Jupiter_1/CloudTopE.dat"\n',  # Output emission cloud top file name
                      #
                      # Cloud Top Determination
                      '#define OptCloudTop          1.0\n',  # Optical Depth of the Cloud Top
                      #
                      '#endif\n',
                      #
                      # 1 Tg yr-1 = 3.7257E+9 H /cm2/s for earth
                      ]
        with open(self.c_code_directory + 'par_' + str(self.process) + '.h', 'w') as file:
            for riga in c_par_file:
                file.write(riga)

    def __core_c_file(self):
        c_core_file = ['#include <stdio.h>\n',
                       '#include <math.h>\n',
                       '#include <stdlib.h>\n',
                       '#include <string.h>\n',
                       #
                       '#include "par_' + str(self.process) + '.h"\n',
                       #
                       '#include "constant.h"\n',
                       '#include "routine.h"\n',
                       '#include "global_rad_gasplanet.h"\n',
                       '#include "GetData.c"\n',
                       '#include "Interpolation.c"\n',
                       '#include "nrutil.h"\n',
                       '#include "nrutil.c"\n',
                       '#include "Convert.c"\n',
                       '#include "TPPara.c"\n',
                       '#include "TPScale.c"\n',
                       '#include "RefIdx.c"\n',
                       '#include "readcross.c"\n',
                       '#include "readcia.c"\n',
                       '#include "Reflection_General_Phase.c"\n',
                       '#include "Trapz.c"\n',
                       #
                       # external (global) variables
                       #
                       'double thickl;\n',
                       'double zl[zbin+1];\n',
                       'double pl[zbin+1];\n',
                       'double tl[zbin+1];\n',
                       'double MM[zbin+1];\n',
                       'double MMZ[zbin+1];\n',
                       'double wavelength[NLAMBDA];\n',
                       'double solar[NLAMBDA];\n',
                       'double crossr[NLAMBDA], crossa[3][NLAMBDA], sinab[3][NLAMBDA], asym[3][NLAMBDA];\n',
                       'double **opacCO2, **opacO2, **opacSO2, **opacH2O, **opacOH, **opacH2CO;\n',
                       'double **opacH2O2, **opacHO2, **opacH2S, **opacCO, **opacO3, **opacCH4; \n',
                       'double **opacNH3;\n',
                       'double **opacC2H2, **opacC2H4, **opacC2H6, **opacHCN, **opacCH2O2, **opacHNO3;\n',
                       'double **opacN2O, **opacN2, **opacNO, **opacNO2, **opacOCS;\n',
                       'double **opacHF, **opacHCl, **opacHBr, **opacHI, **opacClO, **opacHClO;\n',
                       'double **opacHBrO, **opacPH3, **opacCH3Cl, **opacCH3Br, **opacDMS, **opacCS2;\n',
                       'int    ReactionR[NKin+1][7], ReactionM[NKinM+1][5], ReactionP[NPho+1][9], ReactionT[NKinT+1][4];\n',
                       'int    numr=0, numm=0, numt=0, nump=0, numx=0, numc=0, numf=0, numa=0, waternum=0, waterx=0;\n',
                       'double **xx, **xx1, **xx2, **xx3, **xx4;\n',
                       'double TransOptD[zbin+1][NLAMBDA], RefOptD[zbin+1][NLAMBDA];\n',
                       # /*double H2CIA[zbin+1][NLAMBDA], H2HeCIA[zbin+1][NLAMBDA], N2CIA[zbin+1][NLAMBDA], CO2CIA[zbin+1][NLAMBDA];*/\n',
                       'double H2H2CIA[zbin+1][NLAMBDA], H2HeCIA[zbin+1][NLAMBDA], H2HCIA[zbin+1][NLAMBDA], N2H2CIA[zbin+1][NLAMBDA], N2N2CIA[zbin+1][NLAMBDA], CO2CO2CIA[zbin+1][NLAMBDA];\n',
                       'double cH2O[zbin+1][NLAMBDA], cNH3[zbin+1][NLAMBDA], gH2O[zbin+1][NLAMBDA];\n',
                       'double gNH3[zbin+1][NLAMBDA], aH2O[zbin+1][NLAMBDA], aNH3[zbin+1][NLAMBDA]; \n',
                       #
                       'int main()\n',
                       '{\n',
                       '    int s,i,ii,j,jj,jjj,k,nn,qytype,stdnum;\n',
                       '    int nums, numx1=1, numf1=1, numc1=1, numr1=1, numm1=1, nump1=1, numt1=1;\n',
                       '    char *temp;\n',
                       '    char dataline[10000];\n',
                       '    double temp1, wavetemp, crosstemp, DD, GA, mixtemp;\n',
                       '    double z[zbin+1], T[zbin+1], PP[zbin+1], P[zbin+1];\n',
                       '    double *wavep, *crossp, *crosspa, *qyp, *qyp1, *qyp2, *qyp3, *qyp4, *qyp5, *qyp6, *qyp7, **cross, **qy;\n',
                       '    double **crosst, **qyt, *crosspt, *qypt, *qyp1t, *qyp2t, *qyp3t, *qyp4t, *qyp5t, *qyp6t, *qyp7t;\n',
                       '    FILE *fspecies, *fzone, *fhenry, *fp, *fp1, *fp2, *fp3;\n',
                       '    FILE *fout, *fout1, *fout3, *fout4, *fcheck, *ftemp, *fout5, *foutp, *foutc;\n',
                       '    FILE *fimport, *fimportcheck;\n',
                       '    FILE *TPPrint;\n',
                       #
                       '    xx = dmatrix(1,zbin,1,NSP);\n',
                       '    xx1 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx2 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx3 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx4 = dmatrix(1,zbin,1,NSP);\n',
                       #
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=1; i<=NSP; i++) {\n',
                       '            xx[j][i] = 0.0;\n',
                       '            xx1[j][i] = 0.0;\n',
                       '            xx2[j][i] = 0.0;\n',
                       '            xx3[j][i] = 0.0;\n',
                       '            xx4[j][i] = 0.0;\n',
                       '        }\n',
                       '    }\n',
                       #
                       '    GA = ' + str(self.param['gp']) + ';\n',  # Planet Surface Gravity Acceleration, in SI
                       #
                       #    Set the wavelength for calculation
                       '    double dlambda, start, interval, lam[NLAMBDA];\n',
                       '    start = log10(LAMBDALOW);\n',
                       '    interval = log10(LAMBDAHIGH) - log10(LAMBDALOW);\n',
                       '    dlambda = interval / (NLAMBDA-1.0);\n',
                       '    for (i=0; i<NLAMBDA; i++){\n',
                       '        wavelength[i] = pow(10.0, start+i*dlambda)*1.0E+9;\n',  # in nm
                       '        lam[i] = wavelength[i]*1.0E-3;\n',  # in microns
                       '    }\n',
                       #
                       #    Rayleigh Scattering
                       '    double refidx0,DenS;\n',
                       '    DenS = 101325.0 / KBOLTZMANN / 273.0 * 1.0E-6;\n',
                       '    for (i=0; i<NLAMBDA; i++){\n',
                       '        if (RefIdxType == 0) { refidx0=AirRefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 1) { refidx0=CO2RefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 2) { refidx0=HeRefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 3) { refidx0=N2RefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 4) { refidx0=NH3RefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 5) { refidx0=CH4RefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 6) { refidx0=H2RefIdx(wavelength[i]);}\n',
                       '        if (RefIdxType == 7) { refidx0=O2RefIdx(wavelength[i]);}\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr[i]=1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        if (RefIdxType == 6) {crossr[i] = 8.14e-13*pow(wavelength[i]*10.0,-4)+1.28e-6*pow(wavelength[i]*10.0,-6)+1.61*pow(wavelength[i]*10.0,-8); }\n',  # Dalgarno 1962
                       # '\t\t  printf("%s\\t%f\\t%s\\t%e\\n", "The reyleigh scattering cross-section at wavelength", wavelength[i], "nm is", crossr[i]);\n',
                       '    }\n',
                       #
                       # Obtain the stellar radiation
                       '    fp2 = fopen(STAR_SPEC,"r");\n',
                       '    fp3 = fopen(STAR_SPEC,"r");\n',
                       '    s = LineNumber(fp2, 1000);\n',
                       '    double swave[s], sflux[s];\n',
                       '    GetData(fp3, 1000, s, swave, sflux);\n',
                       '    fclose(fp2);\n',
                       '    fclose(fp3);\n',
                       '    Interpolation(wavelength, NLAMBDA, solar, swave, sflux, s, 0);\n',
                       '    for (i=0; i<NLAMBDA; i++) {\n',
                       '        solar[i] = solar[i]/ORBIT/ORBIT*FaintSun;\n',  # convert from flux at 1 AU
                       '    }\n',
                       '    i=0;\n',
                       '    while (solar[i]>0 || wavelength[i]<9990 ) { i++;}\n',
                       '    for (j=i; j<NLAMBDA; j++) {\n',
                       '        solar[j] = solar[i-1]*pow(wavelength[i-1],4)/pow(wavelength[j],4);\n',
                       '    }\n',
                       # '\t  printf("%s\\n", "The stellar radiation data are imported.");\n',
                       #
                       # Import Species List
                       '    fspecies=fopen(SPECIES_LIST, "r");\n',
                       '    s=LineNumber(fspecies, 10000);\n',
                       # '\tprintf("Species list: \\n");\n',
                       '    fclose(fspecies);\n',
                       '    fspecies=fopen(SPECIES_LIST, "r");\n',
                       '    struct Molecule species[s];\n',
                       '    temp=fgets(dataline, 10000, fspecies);\n',  # Read in the header line
                       '    i=0;\n',
                       '    while (fgets(dataline, 10000, fspecies) != NULL )\n',
                       '    {\n',
                       '        sscanf(dataline, "%s %s %d %d %lf %lf %d %lf %lf", (species+i)->name, (species+i)->type, &((species+i)->num), &((species+i)->mass), &((species+i)->mix), &((species+i)->upper), &((species+i)->lowertype), &((species+i)->lower), &((species+i)->lower1));\n',
                       # '\t\tprintf("%s %s %d %d %lf %lf %d %lf %lf\\n",(species+i)->name, (species+i)->type, (species+i)->num, (species+i)->mass, (species+i)->mix, (species+i)->upper, (species+i)->lowertype, (species+i)->lower, (species+i)->lower1);\n',
                       '        if (strcmp("X",species[i].type)==0) {numx=numx+1;}\n',
                       '        if (strcmp("F",species[i].type)==0) {numf=numf+1;}\n',
                       '        if (strcmp("C",species[i].type)==0) {numc=numc+1;}\n',
                       '        if (strcmp("A",species[i].type)==0) {numx=numx+1; numa=numa+1;}\n',
                       '        i=i+1;\n',
                       '    }\n',
                       '    fclose(fspecies);\n',
                       '    nums=numx+numf+numc;\n',
                       # '\tprintf("%s\\n", "The species list is imported.");\n',
                       # '\tprintf("%s %d\\n", "Number of species in model:", nums);\n',
                       # '\tprintf("%s %d\\n", "Number of species to be solved in full:", numx);\n',
                       # '\tprintf("%s %d\\n", "In which the number of aerosol species is:", numa);\n',
                       # '\tprintf("%s %d\\n", "Number of species to be solved in photochemical equil:", numf);\n',
                       # '\tprintf("%s %d\\n", "Number of species assumed to be constant:", numc);\n',
                       '    int labelx[numx+1], labelc[numc+1], labelf[numf+1], MoleculeM[numx+1], listAER[numa+1], AERCount=1;\n',
                       '    for (i=0; i<s; i++) {\t\t\t\n',
                       '        if (strcmp("X",species[i].type)==0 || strcmp("A",species[i].type)==0) {\n',
                       '            labelx[numx1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) { \n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       '            if (species[i].num==7) {\n',
                       '                waternum=numx1;\n',
                       '                waterx=1;\n',
                       '            }\n',
                       '            MoleculeM[numx1]=species[i].mass;\n',
                       '            if (species[i].lowertype==1) {\n',
                       '                xx[1][species[i].num]=species[i].lower1*MM[1];\n',
                       '            }\n',
                       '            if (strcmp("A",species[i].type)==0) {\n',
                       '                listAER[AERCount]=numx1;\n',
                       '                AERCount = AERCount+1;\n',
                       #                printf("%s %d\\n", "The aerosol species is", numx1);\n',
                       '            }\n',
                       '            numx1=numx1+1;\n',
                       '        }\n',
                       '        if (strcmp("F",species[i].type)==0) {\n',
                       '            labelf[numf1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) { \n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       '            numf1=numf1+1;\n',
                       '        }\n',
                       '        if (strcmp("C",species[i].type)==0) {\n',
                       '            labelc[numc1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) {\n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       # import constant mixing ratio list for H2O
                       '            if (IFIMPORTH2O == 1 && species[i].num == 7) {\n',
                       '                fimport=fopen("Data/ConstantMixing.dat", "r");\n',
                       '                fimportcheck=fopen("Data/ConstantMixingH2O.dat", "w");\n',
                       '                temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '                for (j=1; j<=zbin; j++) {\n',
                       '                    fscanf(fimport, "%lf\\t", &temp1);\n',
                       '                    fscanf(fimport, "%le\\t", &mixtemp);\n',
                       '                    fscanf(fimport, "%le\\t", &temp1);\n',
                       '                    xx[j][7]=mixtemp * MM[j];\n',
                       '                    fprintf(fimportcheck, "%f\\t%e\\t%e\\n", zl[j], mixtemp, xx[j][7]);\n',
                       '                }\n',
                       '                fclose(fimport);\n',
                       '                fclose(fimportcheck);\n',
                       '            }\n',
                       # import constant mixing ratio list for CO2
                       '            if (IFIMPORTCO2 == 1 && species[i].num == 52) {\n',
                       '                fimport=fopen("Data/ConstantMixing.dat", "r");\n',
                       '                fimportcheck=fopen("Data/ConstantMixingCO2.dat", "w");\n',
                       '                temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '                for (j=1; j<=zbin; j++) {\n',
                       '                    fscanf(fimport, "%lf\\t", &temp1);\n',
                       '                    fscanf(fimport, "%le\\t", &temp1);\n',
                       '                    fscanf(fimport, "%le\\t", &mixtemp);\n',
                       '                    xx[j][52]=mixtemp * MM[j];\n',
                       '                    fprintf(fimportcheck, "%f\\t%e\\t%e\\n", zl[j], mixtemp, xx[j][52]);\n',
                       '                }\n',
                       '                fclose(fimport);\n',
                       '                fclose(fimportcheck);\n',
                       '            }\n',
                       '            numc1=numc1+1;\n',
                       '        }\n',
                       '    }\n',
                       '    fimport=fopen(IMPORTFILEX, "r");\n',
                       '    fimportcheck=fopen("Data/Fimportcheck.dat","w");\n',
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',
                       '        fprintf(fimportcheck, "%lf\\t", temp1);\n',
                       '        for (i=1; i<=numx; i++) {\n',
                       '            fscanf(fimport, "%le\\t\\t", &xx[j][labelx[i]]);\n',
                       '            fprintf(fimportcheck, "%e\\t", xx[j][labelx[i]]);\n',
                       '        }\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',  # column of air
                       '        fprintf(fimportcheck,"\\n");\n',
                       '    }\n',
                       '    fclose(fimport);\n',
                       '    fimport=fopen(IMPORTFILEF, "r");\n',
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',
                       '        fprintf(fimportcheck, "%lf\\t", temp1);\n',
                       '        for (i=1; i<=numf; i++) {\n',
                       '            fscanf(fimport, "%le\\t\\t", &xx[j][labelf[i]]);\n',
                       '            fprintf(fimportcheck, "%e\\t", xx[j][labelf[i]]);\n',
                       '        }\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',  # column of air
                       '        fprintf(fimportcheck,"\\n");\n',
                       '    }\n',
                       '    fclose(fimport);\n',
                       '    fclose(fimportcheck);\n',
                       #
                       # Set up atmospheric profiles
                       #
                       '    char outstd[1024];\n',
                       '    strcpy(outstd,OUT_DIR);\n',
                       '    strcat(outstd,"ConcentrationSTD.dat");\n',
                       '    if (IMODE == 4) {\n',  # Import the computed profile directly
                       '        fimport=fopen(outstd, "r");\n',
                       '        fimportcheck=fopen("Data/Fimportcheck.dat","w");\n',
                       '        temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '        temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '        for (j=1; j<=zbin; j++) {\n',
                       '            fscanf(fimport, "%lf\\t", &zl[j]);\n',
                       '            fprintf(fimportcheck, "%lf\\t", zl[j]);\n',
                       '            fscanf(fimport, "%lf\\t", &z[j-1]);\n',
                       '            fscanf(fimport, "%lf\\t", &z[j]);\n',
                       '            fscanf(fimport, "%lf\\t", &tl[j]);\n',
                       '            fscanf(fimport, "%le\\t", &pl[j]);\n',
                       '            MM[j] = pl[j]/KBOLTZMANN/tl[j]*1.0E-6;\n',
                       '            for (i=1; i<=NSP; i++) {\n',
                       '                fscanf(fimport, "%le\\t", &xx[j][i]);\n',
                       '                fprintf(fimportcheck, "%e\\t", xx[j][i]);\n',
                       #                MM[j] += xx[j][i];\n',
                       '            }\n',
                       #            printf("%s %f %f\\n", "TP", tl[j], pl[j]);\n',
                       '            fprintf(fimportcheck,"\\n");\n',
                       '        }\n',
                       '        fclose(fimport);\n',
                       '        fclose(fimportcheck);\n',
                       '        thickl = (z[zbin]-z[zbin-1])*1.0E+5;\n',
                       '        for (j=1; j<zbin; j++) {\n',
                       '            T[j] = (tl[j] + tl[j+1])/2.0;\n',
                       '        }\n',
                       '        T[0] = 1.5*tl[1] - 0.5*tl[2];\n',
                       '        T[zbin] = 1.5*tl[zbin] - 0.5*tl[zbin-1];\n',
                       '    }\n',
                       #
                       '    readcia();\n',
                       #
                       # check CIA
                       #    for (i=0; i<NLAMBDA; i++) {\n',
                       #    printf("%s\\t%f\\t%e\\t%e\\t%e\\t%e\\n", "CIA", wavelength[i], H2CIA[1][i], H2HeCIA[1][i], N2CIA[1][i], CO2CIA[1][i]);\n',
                       #    }\n',
                       #
                       # '\tprintf("%s\\n", "Collision-induced absorption cross sections are imported ");\n',
                       #
                       # Obtain the opacity
                       '    opacCO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacSO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacH2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacOH = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacH2CO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacH2O2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacH2S = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacO3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCH4 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacNH3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacC2H2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacC2H4 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacC2H6 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHCN = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCH2O2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHNO3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacN2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacN2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacNO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacNO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacOCS = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHF = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHCl = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHBr = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHI = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacClO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHClO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacHBrO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacPH3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCH3Cl = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCH3Br = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacDMS = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCS2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #
                       '    char crossfile[1024];\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCO2.dat");\n',
                       #    readcross(crossfile, opacCO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacO2.dat");\n',
                       #    readcross(crossfile, opacO2);\n',
                       #
                       '    strcpy(crossfile,CROSSHEADING);\n',
                       '    strcat(crossfile,"opacH2O.dat");\n',
                       '    readcross(crossfile, opacH2O);\n',
                       #
                       '    strcpy(crossfile,CROSSHEADING);\n',
                       '    strcat(crossfile,"opacCH4.dat");\n',
                       '    readcross(crossfile, opacCH4);\n',
                       #
                       '    strcpy(crossfile,CROSSHEADING);\n',
                       '    strcat(crossfile,"opacNH3.dat");\n',
                       '    readcross(crossfile, opacNH3);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCO.dat");\n',
                       #    readcross(crossfile, opacCO);\n',
                       #
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacSO2.dat");\n',
                       #    readcross(crossfile, opacSO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacOH.dat");\n',
                       #    readcross(crossfile, opacOH);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2CO.dat");\n',
                       #    readcross(crossfile, opacH2CO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2O2.dat");\n',
                       #    readcross(crossfile, opacH2O2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHO2.dat");\n',
                       #    readcross(crossfile, opacHO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2S.dat");\n',
                       #    readcross(crossfile, opacH2S);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacO3.dat");\n',
                       #    readcross(crossfile, opacO3);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H2.dat");\n',
                       #    readcross(crossfile, opacC2H2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H4.dat");\n',
                       #    readcross(crossfile, opacC2H4);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H6.dat");\n',
                       #    readcross(crossfile, opacC2H6);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHCN.dat");\n',
                       #    readcross(crossfile, opacHCN);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCH2O2.dat");\n',
                       #    readcross(crossfile, opacCH2O2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHNO3.dat");\n',
                       #    readcross(crossfile, opacHNO3);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacN2O.dat");\n',
                       #    readcross(crossfile, opacN2O);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacN2.dat");\n',
                       #    readcross(crossfile, opacN2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacNO.dat");\n',
                       #    readcross(crossfile, opacNO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacNO2.dat");\n',
                       #    readcross(crossfile, opacNO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacOCS.dat");\n',
                       #    readcross(crossfile, opacOCS);\n',
                       #
                       #    foutc = fopen("Data/IRCross.dat","w");\n',
                       #    for (i=0; i<NLAMBDA; i++) {\n',
                       #        fprintf(foutc, "%f\\t", wavelength[i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2O[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCH4[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNH3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacSO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacOH[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2CO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2O2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2S[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacO3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H4[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H6[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHCN[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCH2O2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHNO3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacN2O[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacN2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacOCS[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHF[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHCl[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHBr[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHI[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacClO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHClO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHBrO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacPH3[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCH3Cl[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCH3Br[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacDMS[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCS2[1][i]);*/\n',
                       #    }\n',
                       #    fclose(foutc);\n',
                       #
                       #    tprintf("%s\\n", "Molecular cross sections are imported ");\n',
                       #
                       # Get Reaction List
                       '    fzone=fopen(REACTION_LIST, "r");\n',
                       '    s=LineNumber(fzone, 10000);\n',
                       '    fclose(fzone);\n',
                       '    fzone=fopen(REACTION_LIST, "r");\n',
                       '    struct Reaction React[s];\n',
                       '    temp=fgets(dataline, 10000, fzone);\n',  # Read in the header line
                       '    i=0;\n',
                       '    while (fgets(dataline, 10000, fzone) != NULL )\n',
                       '    {\n',
                       '        sscanf(dataline, "%d %s %d", &((React+i)->dum), (React+i)->type, &((React+i)->num));\n',
                       #        printf("%d %s %d\\n", (React+i)->dum, React[i].type, React[i].num);\n',
                       '        if (strcmp("R",React[i].type)==0) {numr=numr+1;}\n',
                       '        if (strcmp("M",React[i].type)==0) {numm=numm+1;}\n',
                       '        if (strcmp("P",React[i].type)==0) {nump=nump+1;}\n',
                       '        if (strcmp("T",React[i].type)==0) {numt=numt+1;}\n',
                       '        i=i+1;\n',
                       '    }\n',
                       '    fclose(fzone);\n',
                       '    int zone_r[numr+1], zone_m[numm+1], zone_p[nump+1], zone_t[numt+1];\n',
                       '    for (i=0; i<s; i++) {\n',
                       '        if (strcmp("R",React[i].type)==0) {\n',
                       '            zone_r[numr1]=React[i].num;\n',
                       '            numr1=numr1+1;\n',
                       '        }\n',
                       '        if (strcmp("M",React[i].type)==0) {\n',
                       '            zone_m[numm1]=React[i].num;\n',
                       '            numm1=numm1+1;\n',
                       '        }\n',
                       '        if (strcmp("P",React[i].type)==0) {\n',
                       '            zone_p[nump1]=React[i].num;\n',
                       '            nump1=nump1+1;\n',
                       '        }\n',
                       '        if (strcmp("T",React[i].type)==0) {\n',
                       '            zone_t[numt1]=React[i].num;\n',
                       '        numt1=numt1+1;\n',
                       '        }\n',
                       '    }\n',
                       #    printf("%s\\n", "The reaction lists are imported.");\n',
                       #    tprintf("%s %d\\n", "Number of bi-molecular reactions:", numr);\n',
                       #    tprintf("%s %d\\n", "Number of tri-molecular reactions:", numm);\n',
                       #    tprintf("%s %d\\n", "Number of photolysis:", nump);\n',
                       #    tprintf("%s %d\\n", "Number of thermo-dissociations:", numt);\n',
                       '    GetReaction();\n',
                       #
                       # get the cross sections and quantum yields of molecules
                       '    cross=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    crosst=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    qy=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    qyt=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    int stdcross[nump+1];\n',
                       '    double qysum[nump+1];\n',
                       #    fcheck=fopen("Data/CrossSectionCheck.dat","w"); \n',
                       '    for (i=1; i<=nump; i++) {\n',
                       '        stdcross[i]=ReactionP[zone_p[i]][1];\n',
                       '        qytype=ReactionP[zone_p[i]][8];\n',
                       '        qysum[i]=ReactionP[zone_p[i]][7];\n',
                       '        j=0;\n',
                       '        while (species[j].num != stdcross[i]) {j=j+1;}\n',
                       #        printf("%s\\n",species[j].name);\n',
                       '        fp=fopen(species[j].name, "r");\n',
                       '        fp1=fopen(species[j].name, "r");\n',
                       '        s=LineNumber(fp, 1000);\n',
                       #        printf("%d\\n",s);\n',
                       '        wavep=dvector(0,s-1);\n',
                       '        crossp=dvector(0,s-1);\n',
                       '        qyp=dvector(0,s-1);\n',
                       '        qyp1=dvector(0,s-1);\n',
                       '        qyp2=dvector(0,s-1);\n',
                       '        qyp3=dvector(0,s-1);\n',
                       '        qyp4=dvector(0,s-1);\n',
                       '        qyp5=dvector(0,s-1);\n',
                       '        qyp6=dvector(0,s-1);\n',
                       '        qyp7=dvector(0,s-1);\n',
                       '        crosspt=dvector(0,s-1);\n',
                       '        qypt=dvector(0,s-1);\n',
                       '        qyp1t=dvector(0,s-1);\n',
                       '        qyp2t=dvector(0,s-1);\n',
                       '        qyp3t=dvector(0,s-1);\n',
                       '        qyp4t=dvector(0,s-1);\n',
                       '        qyp5t=dvector(0,s-1);\n',
                       '        qyp6t=dvector(0,s-1);\n',
                       '        qyp7t=dvector(0,s-1);\n',
                       '        k=0;\n',
                       '        if (qytype==1) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf", wavep+k, crossp+k, crosspt+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==2) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==3) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==4) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==5) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==6) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==7) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp6+k, qyp6t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==8) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp6+k, qyp6t+k, qyp7+k, qyp7t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        fclose(fp);\n',
                       '        fclose(fp1);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(cross+i), wavep, crossp, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(qy+i), wavep, qyp, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(crosst+i), wavep, crosspt, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(qyt+i), wavep, qypt, s, 0);\n',
                       '        free_dvector(wavep,0,s-1);\n',
                       '        free_dvector(crossp,0,s-1);\n',
                       '        free_dvector(qyp,0,s-1);\n',
                       '        free_dvector(qyp1,0,s-1);\n',
                       '        free_dvector(qyp2,0,s-1);\n',
                       '        free_dvector(qyp3,0,s-1);\n',
                       '        free_dvector(qyp4,0,s-1);\n',
                       '        free_dvector(qyp5,0,s-1);\n',
                       '        free_dvector(qyp6,0,s-1);\n',
                       '        free_dvector(qyp7,0,s-1);\n',
                       '        free_dvector(crosspt,0,s-1);\n',
                       '        free_dvector(qypt,0,s-1);\n',
                       '        free_dvector(qyp1t,0,s-1);\n',
                       '        free_dvector(qyp2t,0,s-1);\n',
                       '        free_dvector(qyp3t,0,s-1);\n',
                       '        free_dvector(qyp4t,0,s-1);\n',
                       '        free_dvector(qyp5t,0,s-1);\n',
                       '        free_dvector(qyp6t,0,s-1);\n',
                       '        free_dvector(qyp7t,0,s-1);\n',
                       #        printf("%s %s %s\\n", "The", species[j].name, "Cross section and quantum yield data are imported.");\n',
                       #        fprintf(fcheck, "%s %s %s\\n", "The", species[j].name, "Cross section and quantum yield data are imported.");\n',
                       #        for (j=0; j<NLAMBDA;j++) {fprintf(fcheck, "%lf %le %le %lf %lf\\n", wavelength[j], cross[i][j], crosst[i][j], qy[i][j], qyt[i][j]);}\n',
                       '    }\n',
                       #
                       # cross section of aerosols
                       '    double *crossp1, *crossp2, *crossp3;\n',
                       '    double crossw1[NLAMBDA], crossw2[NLAMBDA], crossw3[NLAMBDA];\n',
                       '    fp=fopen(AERRADFILE1,"r");\n',
                       '    fp1=fopen(AERRADFILE1,"r");\n',
                       '    s=LineNumber(fp, 1000);\n',
                       '    wavep=dvector(0,s-1);\n',
                       '    crossp1=dvector(0,s-1);\n',
                       '    crossp2=dvector(0,s-1);\n',
                       '    crossp3=dvector(0,s-1);\n',
                       '    k=0;\n',
                       '    while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '        sscanf(dataline, "%lf %lf %lf %lf", wavep+k, crossp1+k, crossp2+k, crossp3+k);\n',
                       '        k=k+1; \n',
                       '    }\n',
                       '    fclose(fp);\n',
                       '    fclose(fp1);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw1, wavep, crossp1, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw2, wavep, crossp2, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw3, wavep, crossp3, s, 0);\n',
                       '    free_dvector(wavep,0,s-1);\n',
                       '    free_dvector(crossp1,0,s-1);\n',
                       '    free_dvector(crossp2,0,s-1);\n',
                       '    free_dvector(crossp3,0,s-1);\n',
                       '    for (i=0; i<NLAMBDA; i++) {\n',
                       '        crossa[1][i] = crossw1[i];\n',
                       '        sinab[1][i]  = crossw2[i]/(crossw1[i]+1.0e-24);\n',
                       '        asym[1][i]   = crossw3[i];\n',
                       '    }\n',
                       '    fp=fopen(AERRADFILE2,"r");\n',
                       '    fp1=fopen(AERRADFILE2,"r");\n',
                       '    s=LineNumber(fp, 1000);\n',
                       '    wavep=dvector(0,s-1);\n',
                       '    crossp1=dvector(0,s-1);\n',
                       '    crossp2=dvector(0,s-1);\n',
                       '    crossp3=dvector(0,s-1);\n',
                       '    k=0;\n',
                       '    while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '        sscanf(dataline, "%lf %lf %lf %lf", wavep+k, crossp1+k, crossp2+k, crossp3+k);\n',
                       '        k=k+1; \n',
                       '    }\n',
                       '    fclose(fp);\n',
                       '    fclose(fp1);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw1, wavep, crossp1, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw2, wavep, crossp2, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw3, wavep, crossp3, s, 0);\n',
                       '    free_dvector(wavep,0,s-1);\n',
                       '    free_dvector(crossp1,0,s-1);\n',
                       '    free_dvector(crossp2,0,s-1);\n',
                       '    free_dvector(crossp3,0,s-1);\n',
                       '    for (i=0; i<NLAMBDA; i++) {\n',
                       '        crossa[2][i] = crossw1[i];\n',
                       '        sinab[2][i]  = crossw2[i]/(crossw1[i]+1.0e-24);\n',
                       '        asym[2][i]   = crossw3[i];\n',
                       '    }\n',
                       #    printf("%s\\n", "Cross sections of the aerosol are imported.");\n',
                       #    fprintf(fcheck, "%s\\n", "Cross sections of the aerosol are imported.");\n',
                       #    for (j=0; j<NLAMBDA;j++) {fprintf(fcheck, "%lf %e %e %f %f %f %f\\n", wavelength[j], crossa[1][j], crossa[2][j], sinab[1][j], sinab[2][j], asym[1][j], asym[2][j]);}\n',
                       #    fclose(fcheck);\n',
                       #
                       #
                       '    FILE *fim;\n',
                       '    double lll[5]={450.0,550.0,675.0,825.0,975.0}, ccc[5];\n',
                       #
                       '    char outaer1[1024];\n',
                       '    strcpy(outaer1,OUT_DIR);\n',
                       '    strcat(outaer1,"cross_H2O.dat");\n',
                       '    fim=fopen(outaer1,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i<5; i++) { fscanf(fim, "%le", ccc+i); }\n',
                       '        for (i=0; i<NLAMBDA; i++) {\n',
                       '            Interpolation(&wavelength[i], 1, &cH2O[j][i], lll, ccc, 5, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',
                       #
                       '    char outaer2[1024];\n',
                       '    strcpy(outaer2,OUT_DIR);\n',
                       '    strcat(outaer2,"cross_NH3.dat");\n',
                       '    fim=fopen(outaer2,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i<5; i++) { fscanf(fim, "%le", ccc+i); }\n',
                       '        for (i=0; i<NLAMBDA; i++) {\n',
                       '            Interpolation(&wavelength[i], 1, &cNH3[j][i], lll, ccc, 5, 2);\n',
                       '        }\n',
                       #        printf("%e\\t%2.6e\\t%2.6e\\n",pl[j],cH2O[j][3650],cNH3[j][3650]);\n',
                       '    }\n',
                       '    fclose(fim);\n',
                       #
                       '    char outaer3[1024];\n',
                       '    strcpy(outaer3,OUT_DIR);\n',
                       '    strcat(outaer3,"geo_H2O.dat");\n',
                       '    fim=fopen(outaer3,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i<5; i++) { fscanf(fim, "%lf", ccc+i); }\n',
                       '        for (i=0; i<NLAMBDA; i++) {\n',
                       '            Interpolation(&wavelength[i], 1, &gH2O[j][i], lll, ccc, 5, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',
                       #
                       '    char outaer4[1024];\n',
                       '    strcpy(outaer4,OUT_DIR);\n',
                       '    strcat(outaer4,"geo_NH3.dat");\n',
                       '    fim=fopen(outaer4,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i<5; i++) { fscanf(fim, "%lf", ccc+i); }\n',
                       '        for (i=0; i<NLAMBDA; i++) {\n',
                       '            Interpolation(&wavelength[i], 1, &gNH3[j][i], lll, ccc, 5, 2);\n',
                       '        }\n',
                       #        printf("%e\\t%2.6f\\t%2.6f\\n",pl[j],gH2O[j][3650],gNH3[j][3650]);\n',
                       '    }\n',
                       '    fclose(fim);\n',
                       #
                       '    char outaer5[1024];\n',
                       '    strcpy(outaer5,OUT_DIR);\n',
                       '    strcat(outaer5,"albedo_H2O.dat");\n',
                       '    fim=fopen(outaer5,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i<5; i++) { fscanf(fim, "%lf", ccc+i); }\n',
                       '        for (i=0; i<NLAMBDA; i++) {\n',
                       '            Interpolation(&wavelength[i], 1, &aH2O[j][i], lll, ccc, 5, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',
                       #
                       '    char outaer6[1024];\n',
                       '    strcpy(outaer6,OUT_DIR);\n',
                       '    strcat(outaer6,"albedo_NH3.dat");\n',
                       '    fim=fopen(outaer6,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i<5; i++) { fscanf(fim, "%lf", ccc+i); }\n',
                       '        for (i=0; i<NLAMBDA; i++) {\n',
                       '            Interpolation(&wavelength[i], 1, &aNH3[j][i], lll, ccc, 5, 2);\n',
                       '        }\n',
                       #        printf("%e\\t%2.6f\\t%2.6f\\n",pl[j],aH2O[j][3650],aNH3[j][3650]);\n',
                       '    }\n',
                       '    fclose(fim);\n',
                       #
                       # Geometric Albedo 9-point Gauss Quadruture
                       '    double cmiu[9]={-0.9681602395076261,-0.8360311073266358,-0.6133714327005904,-0.3242534234038089,0.0,0.3242534234038089,0.6133714327005904,0.8360311073266358,0.9681602395076261};\n',
                       '    double wmiu[9]={0.0812743883615744,0.1806481606948574,0.2606106964029354,0.3123470770400029,0.3302393550012598,0.3123470770400029,0.2606106964029354,0.1806481606948574,0.0812743883615744};\n',
                       #
                       '    double phase;\n',
                       '    phase = ' + str(self.param['phi']) + ';\n',  # Phase Angle, 0 zero geometric albedo
                       '    double lonfactor1, lonfactor2;\n',
                       '    double latfactor1, latfactor2;\n',
                       '    lonfactor1 = (PI-phase)*0.5;\n',
                       '    lonfactor2 = phase*0.5;\n',
                       '    latfactor1 = PI*0.5;\n',
                       '    latfactor2 = 0;\n',
                       #
                       '    double lat[9], lon[9];\n',
                       '    for (i=0; i<9; i++) {\n',
                       '        lat[i] = latfactor1*cmiu[i]+latfactor2;\n',
                       '        lon[i] = lonfactor1*cmiu[i]+lonfactor2;\n',
                       '    }\n',
                       #
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=1; i<=NSP; i++) {\n',
                       '            xx1[j][i] = xx[j][i];\n',
                       '        }\n',
                       '    }\n',
                       #
                       '    double gmiu0, gmiu;\n',
                       '    double rout[NLAMBDA],gal[NLAMBDA];\n',
                       '    for (k=0; k<NLAMBDA; k++) {\n',
                       '        gal[k] = 0;\n',
                       '    }\n',
                       #
                       '    double T0[zbin+1];\n',
                       '    for (j=0; j<=zbin; j++) {\n',
                       '        T0[j]=0.0;\n',
                       '    }\n',
                       #
                       '    char uvrfile[1024];\n',
                       '    strcpy(uvrfile,OUT_DIR);\n',
                       '    strcat(uvrfile,"Reflection_Phase.dat");\n',
                       '    for (i=0; i<9; i++) {\n',
                       '        for (j=0; j<9; j++) {\n',
                       '            gmiu0 = cos(lat[i])*cos(lon[j]-phase);\n',
                       '            gmiu  = cos(lat[i])*cos(lon[j]);\n',
                       '            if (fabs(gmiu0-gmiu)<0.0000001) {\n',
                       '                gmiu=gmiu0+0.0000001;\n',
                       '            }\n',
                       # '            printf("%f %f %f %f\\n", lat[i], lon[j], gmiu0, gmiu);\n',
                       '            Reflection(xx1 , T0, stdcross, qysum, cross, crosst, uvrfile, gmiu0, gmiu, phase, rout);\n',
                       '            for (k=0; k<NLAMBDA; k++) {\n',
                       '                gal[k] += wmiu[i]*wmiu[j]*rout[k]*gmiu0*gmiu*cos(lat[i])*latfactor1*lonfactor1/PI;\n',
                       '            }\n',
                       '        }\n',
                       '    }\n',
                       #
                       # Variation
                       #
                       # '    char ANGLE[][1024]={"G1","G2","G3","G4","G5","G6","G7","G8"};\n',
                       # '    char uvrfile[1024];\n',
                       #
                       # '    double methaneexp[5]={0,0,0,0,0};\n',
                       # '    int methaneid;\n',
                       #
                       # '    double rout[NLAMBDA],gal[NLAMBDA][5];\n',
                       #
                       # '    for (j=0; j<NLAMBDA; j++) {\n',
                       # '        for (methaneid=0; methaneid<5; methaneid++) {\n',
                       # '            gal[j][methaneid]=0;\n',
                       # '        }\n',
                       # '    }\n',
                       # '    for (methaneid=0; methaneid<5; methaneid++) {\n',
                       #
                       # '        for (j=1; j<=zbin; j++) {\n',
                       # '            for (i=1; i<=NSP; i++) {\n',
                       # '                xx1[j][i] = xx[j][i];\n',
                       # '            }\n',
                       # '            if (methaneid==1) { CH4 only\n',
                       # '                xx1[j][20] = 0;\n',
                       # '                xx1[j][9] = 0;\n',
                       # '                xx1[j][7] = 0;\n',
                       # '            }\n',
                       # '            if (methaneid==2) { CO only\n',
                       # '                xx1[j][21] = 0;\n',
                       # '                xx1[j][9] = 0;\n',
                       # '                xx1[j][7] = 0;\n',
                       # '            }\n',
                       # '            if (methaneid==3) { H2O only\n',
                       # '                xx1[j][21] = 0;\n',
                       # '                xx1[j][20] = 0;\n',
                       # '                xx1[j][9] = 0;\n',
                       # '            }\n',
                       # '            if (methaneid==4) { NH3 only\n',
                       # '                xx1[j][21] = 0;\n',
                       # '                xx1[j][20] = 0;\n',
                       # '                xx1[j][7] = 0;\n',
                       # '            }\n',
                       # '        }\n',
                       #
                       # '        for (i=0; i<8; i++) {\n',
                       # '            strcpy(uvrfile,OUT_DIR);\n',
                       # '            strcat(uvrfile,"Reflection_");\n',
                       # '            strcat(uvrfile,ANGLE[i]);\n',
                       # '            strcat(uvrfile,".dat");\n',
                       # '            Reflection(xx1 , T, stdcross, qysum, cross, crosst, uvrfile, gmiu[i], gmiu[i]+0.0000001, rout);\n',
                       # '            for (j=0; j<NLAMBDA; j++) {\n',
                       # '                gal[j][methaneid] += wmiu[i]*rout[j]*gmiu[i]*gmiu[i];\n',
                       # '            }\n',
                       # '        }\n',
                       # '    }\n',
                       #
                       # print out spectra
                       '    char outag[1024];\n',
                       '    strcpy(outag,OUT_DIR);\n',
                       '    strcat(outag,"PhaseA.dat");\n',
                       '    fp=fopen(outag, "w");\n',
                       '    for (i=0; i<NLAMBDA; i++) {\n',
                       '        if (wavelength[i]>100.0 && wavelength[i]<200000.0) {\n',
                       '            fprintf(fp, "%f\\t", wavelength[i]);\n',
                       '            fprintf(fp, "%e\\t", gal[i]);\n',
                       # '          for (methaneid=0; methaneid<5; methaneid++) {\n',
                       # '              fprintf(fp, "%e\\t", gal[i][methaneid]);\n',
                       # '          }\n',
                       '            fprintf(fp, "\\n");\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fp);\n',
                       #
                       # clean up
                       '    free_dmatrix(cross,1,nump,0,NLAMBDA-1);\n',
                       '    free_dmatrix(qy,1,nump,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCO2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacO2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacSO2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacH2O,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacOH,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacH2CO,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacH2O2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHO2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacH2S,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCO,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacO3,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCH4,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacNH3,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacC2H2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacC2H4,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacC2H6,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHCN,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCH2O2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHNO3,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacN2O,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacN2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacNO,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacNO2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacOCS,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHF,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHCl,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHBr,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHI,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacClO,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHClO,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacHBrO,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacPH3,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCH3Cl,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCH3Br,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacDMS,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(opacCS2,1,zbin,0,NLAMBDA-1);\n',
                       '    free_dmatrix(xx,1,zbin,1,NSP);\n',
                       '    free_dmatrix(xx1,1,zbin,1,NSP);\n',
                       '    free_dmatrix(xx2,1,zbin,1,NSP);\n',
                       '    free_dmatrix(xx3,1,zbin,1,NSP);\n',
                       '    free_dmatrix(xx4,1,zbin,1,NSP);\n',
                       #
                       '}\n']

        with open(self.c_code_directory + 'core_' + str(self.process) + '.c', 'w') as file:
            for riga in c_core_file:
                file.write(riga)

    def __run_c_code(self):
        self.__par_c_file()
        self.__core_c_file()
        os.chdir(self.c_code_directory)
        if platform.system() == 'Darwin':
            os.system('clang -o ' + str(self.process) + ' core_' + str(self.process) + '.c -lm')
        else:
            os.system('gcc -o ' + str(self.process) + ' core_' + str(self.process) + '.c -lm')
        while not os.path.exists(self.c_code_directory + str(self.process)):
            pass
        os.system('chmod +rwx ' + str(self.process))
        os.system('./' + str(self.process))
        os.chdir(self.working_dir)
        while not os.path.exists(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/PhaseA.dat'):
            pass
        os.system('rm ' + self.c_code_directory + str(self.process))
        os.system('rm ' + self.c_code_directory + 'core_' + str(self.process) + '.c')
        os.system('rm ' + self.c_code_directory + 'par_' + str(self.process) + '.h')
        albedo = np.loadtxt(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/PhaseA.dat')
        if self.retrieval:
            os.system('rm -rf ' + self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
        else:
            if self.canc_metadata:
                os.system('rm -rf ' + self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
            else:
                pass

        return albedo[:, 0], albedo[:, 1]

    def run_forward(self):
        self.__run_structure()
        alb_wl, alb = self.__run_c_code()

        return alb_wl, alb


class FORWARD_ROCKY_MODEL:
    def __init__(self, param, retrieval=True, canc_metadata=False):
        self.param = copy.deepcopy(param)
        self.process = str(self.param['core_number']) + str(random.randint(0, 100000)) + alphabet() + alphabet() + alphabet() + str(random.randint(0, 100000))
        self.package_dir = param['pkg_dir']
        self.retrieval = retrieval
        self.canc_metadata = canc_metadata
        self.hazes_calc = param['hazes']
        self.c_code_directory = self.package_dir + 'forward_rocky_mod/'
        self.matlab_code_directory = self.c_code_directory + 'PlanetModel/'
        try:
            self.working_dir = param['wkg_dir']
        except KeyError:
            self.working_dir = os.getcwd()

    def __surface_structure(self):
        if self.param['fit_ag']:
            self.surf_alb = np.ones(len(self.param['wl_C_grid']))
            if self.param['surface_albedo_parameters'] == int(1):
                self.surf_alb *= self.param['Ag']
            elif self.param['surface_albedo_parameters'] == int(3):
                x1_indx = np.where(self.param['wl_C_grid'] < self.param['Ag_x1'])[0]
                self.surf_alb[x1_indx] *= self.param['Ag1']
                self.surf_alb[x1_indx[-1] + 1:] *= self.param['Ag2']
            elif self.param['surface_albedo_parameters'] == int(5):
                x1_indx = np.where(self.param['wl_C_grid'] < self.param['Ag_x1'])[0]
                x2_indx = np.where((self.param['wl_C_grid'] > self.param['Ag_x1']) & (self.param['wl_C_grid'] < self.param['Ag_x2']))[0]
                self.surf_alb[x1_indx] *= self.param['Ag1']
                self.surf_alb[x2_indx] *= self.param['Ag2']
                self.surf_alb[x2_indx[-1] + 1:] *= self.param['Ag3']
        else:
            if self.param['Ag'] is not None:
                self.surf_alb = self.param['Ag'] * np.ones(len(self.param['wl_C_grid']))
            else:
                self.surf_alb = np.zeros(len(self.param['wl_C_grid']))

        with open(self.outdir + 'surface_albedo.dat', 'w') as file:
            for i in range(0, len(self.surf_alb)):
                file.write("{:.6e}".format(self.surf_alb[i]))
                file.write('\n')

    def __atmospheric_structure(self):
        try:
            os.mkdir(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
        except OSError:
            self.process = alphabet() + str(random.randint(0, 100000))
            os.mkdir(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')

        self.outdir = self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/'

        deltaP = 0.001 * self.__waterpressure(220)  # assume super saturation to be 0.1% at 220 K

        g = self.param['gp'] + 0.0

        # Set up pressure grid
        P = self.param['P'] + 0.0  # in Pascal

        # Temperature profile (isothermal)
        T = self.param['Tp'] * np.ones(len(P))

        # Cloud density calculation
        cloudden = 1.0e-36 * np.ones(len(P))
        for i in range(len(P) - 2, -1, -1):
            cloudden[i] = max(abs(self.param['vmr_H2O'][i] - self.param['vmr_H2O'][i + 1]) * 0.018 * P[i] / const.R.value / T[i], 1e-25)  # kg/m^3, g/L

        # Particle size calculation
        particlesize = 1.0e-36 * np.ones(len(P))
        if self.param['fit_p_size'] and self.param['p_size_type'] == 'constant':
            particlesize = self.param['p_size'] * np.ones(len(P))
        else:
            for i in range(len(P) - 2, -1, -1):
                r0, r1, r2, VP = particlesizef(g, T[i], P[i], self.param['mean_mol_weight'][i], self.param['mm']['H2O'], self.param['KE'], deltaP)
                if self.param['fit_p_size'] and self.param['p_size_type'] == 'factor':
                    particlesize[i] = r2 * self.param['p_size']
                else:
                    particlesize[i] = r2 + 0.0

        # Calculate the height
        P = P[::-1]
        T = T[::-1]
        cloudden = cloudden[::-1]
        particlesize = particlesize[::-1]
        MMM = self.param['mean_mol_weight'][::-1]

        # Atmospheric Composition
        f = {}
        for mol in self.param['fit_molecules']:
            f[mol] = self.param['vmr_' + mol][::-1]
        if self.param['gas_fill'] is not None:
            f[self.param['gas_fill']] = self.param['vmr_' + self.param['gas_fill']][::-1]

        Z = np.zeros(len(P))
        for j in range(0, len(P) - 1):
            H = const.k_B.value * (T[j] + T[j + 1]) / 2. / g / MMM[j] / const.u.value / 1000.  # km
            Z[j + 1] = Z[j] + H * np.log(P[j] / P[j + 1])

        # Adaptive grid
        if self.param['use_adaptive_grid']:
            idx_cloud_layers = np.where(np.diff(f['H2O']) != 0.0)[0] + 1
            if len(idx_cloud_layers) > 0:
                n_cloud_layers = int(round((self.param['n_layer'] + 1) / 3, 0))
                n_above_layers = int(round((self.param['n_layer'] + 1 - n_cloud_layers) / 2, 0))
                n_below_layers = (self.param['n_layer'] + 1) - n_cloud_layers - n_above_layers

                Z_below = np.linspace(Z[0], Z[min(idx_cloud_layers) - 1], num=n_below_layers, endpoint=False)
                Z_within = np.linspace(Z[min(idx_cloud_layers) - 1], Z[max(idx_cloud_layers)], num=n_cloud_layers, endpoint=False)
                Z_above = np.linspace(Z[max(idx_cloud_layers)], Z[-1], num=n_above_layers, endpoint=True)
                zz = np.concatenate((np.concatenate((Z_below, Z_within)), Z_above))
            else:
                zz = np.linspace(Z[0], Z[-1], num=int(self.param['n_layer'] + 1), endpoint=True)
        else:
            zz = np.linspace(Z[0], Z[-1], num=int(self.param['n_layer'] + 1), endpoint=True)

        if not self.retrieval:
            np.savetxt(self.outdir + 'watermix.dat', f['H2O'])

            np.savetxt(self.outdir + 'particlesize.dat', particlesize)

            np.savetxt(self.outdir + 'cloudden.dat', cloudden)

            np.savetxt(self.outdir + 'P.dat', P)
            np.savetxt(self.outdir + 'T.dat', T)

        z0 = zz[:-1]
        z1 = zz[1:]
        zl = np.mean([z0, z1], axis=0)
        tck = interp1d(Z, T)
        tl = tck(zl)
        tck = interp1d(Z, np.log(P))
        pl = np.exp(tck(zl))

        nden = pl / const.k_B.value / tl * 1.0E-6  # molecule cm-3
        n = {}
        for mol in self.param['fit_molecules']:
            tck = interp1d(Z, np.log(f[mol]))
            n[mol] = np.exp(tck(zl)) * nden
        if self.param['gas_fill'] is not None:
            tck = interp1d(Z, np.log(f[self.param['gas_fill']]))
            n[self.param['gas_fill']] = np.exp(tck(zl)) * nden

        tck = interp1d(Z, np.log(cloudden))
        cloudden = np.exp(tck(zl))

        tck = interp1d(Z, np.log(particlesize))
        particlesize = np.exp(tck(zl))

        #    Generate ConcentrationSTD.dat file
        NSP = 111
        with open(self.outdir + 'ConcentrationSTD.dat', 'w') as file:
            file.write('z\t\tz0\t\tz1\t\tT\t\tP')
            for i in range(1, NSP + 1):
                file.write('\t\t' + str(i))
            file.write('\n')
            file.write('km\t\tkm\t\tkm\t\tK\t\tPa\n')
            for j in range(0, len(zl)):
                file.write("{:.6f}".format(zl[j]) + '\t\t' + "{:.6f}".format(z0[j]) + '\t\t' + "{:.6f}".format(z1[j]) + '\t\t' + "{:.6f}".format(tl[j]) + '\t\t' + "{:.6e}".format(pl[j]))
                for i in range(1, NSP + 1):
                    # H2O
                    if i == 7 and 'H2O' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'H2O':
                            file.write('\t\t' + "{:.6e}".format(n['H2O'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'H2O':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['H2O'][j]))

                    # NH3
                    elif i == 9 and 'NH3' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'NH3':
                            file.write('\t\t' + "{:.6e}".format(n['NH3'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'NH3':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['NH3'][j]))

                    # CH4
                    elif i == 21 and 'CH4' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'CH4':
                            file.write('\t\t' + "{:.6e}".format(n['CH4'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'CH4':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['CH4'][j]))

                    # SO2
                    elif i == 43 and 'SO2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'SO2':
                            file.write('\t\t' + "{:.6e}".format(n['SO2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'SO2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['SO2'][j]))

                    # H2S
                    elif i == 45 and 'H2S' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'H2S':
                            file.write('\t\t' + "{:.6e}".format(n['H2S'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'H2S':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['H2S'][j]))

                    # CO2
                    elif i == 52 and 'CO2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'CO2':
                            file.write('\t\t' + "{:.6e}".format(n['CO2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'CO2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['CO2'][j]))

                    # CO
                    elif i == 20 and 'CO' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'CO':
                            file.write('\t\t' + "{:.6e}".format(n['CO'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'CO':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['CO'][j]))

                    # O2
                    elif i == 54 and 'O2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'O2':
                            file.write('\t\t' + "{:.6e}".format(n['O2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'O2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['O2'][j]))

                    # O3
                    elif i == 2 and 'O3' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'O3':
                            file.write('\t\t' + "{:.6e}".format(n['O3'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'O3':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['O3'][j]))

                    # N2O
                    elif i == 11 and 'N2O' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'N2O':
                            file.write('\t\t' + "{:.6e}".format(n['N2O'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'N2O':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['N2O'][j]))

                    # N2
                    elif i == 55 and 'N2' in self.param['fit_molecules']:
                        if self.param['contribution'] and self.param['mol_contr'] == 'N2':
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'N2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))
                    elif i == 55 and self.param['gas_fill'] == 'N2':
                        if self.param['contribution'] and self.param['mol_contr'] == 'N2':
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'N2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['N2'][j]))

                    # H2
                    elif i == 53 and self.param['gas_fill'] == 'H2':
                        if self.param['contribution'] and self.param['mol_contr'] == 'H2':
                            file.write('\t\t' + "{:.6e}".format(n['H2'][j]))
                        elif self.param['contribution'] and self.param['mol_contr'] != 'H2':
                            file.write('\t\t' + "{:.6e}".format(0.0))
                        else:
                            file.write('\t\t' + "{:.6e}".format(n['H2'][j]))
                    else:
                        file.write('\t\t' + "{:.6e}".format(0.0))
                file.write('\n')

        #    cloud output
        crow = np.zeros((len(zl), 324))
        albw = np.ones((len(zl), 324))
        geow = np.zeros((len(zl), 324))

        #    opacity
        sig = 2
        for j in range(0, len(zl)):
            r2 = particlesize[j]
            if cloudden[j] < 1e-16:
                pass
            else:
                r0 = r2 * np.exp(-np.log(sig) ** 2.)
                VP = 4. * math.pi / 3. * ((r2 * 1.0e-6 * np.exp(0.5 * np.log(sig) ** 2.)) ** 3.) * 1.0e+6 * 1.0  # g
                for indi in range(0, 324):
                    tck = interp1d(np.log10(self.param['H2OL_r']), np.log10(self.param['H2OL_c'][:, indi]))
                    temporaneo = tck(np.log10(max(0.01, min(r0, 100))))
                    crow[j, indi] = cloudden[j] / VP * 1.0e-3 * (10. ** temporaneo)  # cm-1
                    tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_a'][:, indi])
                    albw[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))
                    tck = interp1d(np.log10(self.param['H2OL_r']), self.param['H2OL_g'][:, indi])
                    geow[j, indi] = tck(np.log10(max(0.01, min(r0, 100))))

        with open(self.outdir + 'cross_H2O.dat', 'w') as file:
            for j in range(0, len(zl)):
                for indi in range(0, 324):
                    file.write("{:.6e}".format(crow[j, indi]) + '\t')
                file.write('\n')

        with open(self.outdir + 'albedo_H2O.dat', 'w') as file:
            for j in range(0, len(zl)):
                for indi in range(0, 324):
                    file.write("{:.6e}".format(albw[j, indi]) + '\t')
                file.write('\n')

        with open(self.outdir + 'geo_H2O.dat', 'w') as file:
            for j in range(0, len(zl)):
                for indi in range(0, 324):
                    file.write("{:.6e}".format(geow[j, indi]) + '\t')
                file.write('\n')

    def __waterpressure(self, t):
        # Saturation Vapor Pressure of Water
        # t in K
        # p in Pascal

        try:
            p = np.empty((len(t)))
        except TypeError:
            t = np.array([t])
            p = np.empty(len(t))

        for i in range(0, len(t)):
            if t[i] < 273.16:
                # Formulation from Murphy & Koop(2005)
                p[i] = np.exp(9.550426 - (5723.265 / t[i]) + (3.53068 * np.log(t[i])) - (0.00728332 * t[i]))
            elif t[i] < 373.15:
                # Formulation] from Seinfeld & Pandis(2006)
                a = 1 - (373.15 / t[i])
                p[i] = 101325. * np.exp((13.3185 * a) - (1.97 * (a ** 2.)) - (0.6445 * (a ** 3.)) - (0.1229 * (a ** 4.)))
            elif t[i] < 647.09:
                p[i] = (10. ** (8.14019 - (1810.94 / (244.485 + t[i] - 273.15)))) * 133.322387415
            else:
                p[i] = np.nan
        return p

    def __run_structure(self):
        os.chdir(self.matlab_code_directory)
        self.__atmospheric_structure()
        self.__surface_structure()
        os.chdir(self.working_dir)

    def __par_c_file(self):
        c_par_file = ['#ifndef _PLANET_H_\n',
                      '#define _PLANET_H_\n',
                      # Planet Physical Properties
                      # '#define MASS_PLANET          ' + str(self.param['Mp'] * const.M_jup.value) + '\n',  # kg
                      # '#define RADIUS_PLANET        ' + str(self.param['Rp'] * const.R_jup.value) + '\n',  # m

                      # Planet Orbital Properties
                      '#define ORBIT                ' + str(self.param['equivalent_a']) + '\n',  # AU
                      '#define STAR_SPEC            "Data/solar0.txt"\n',
                      '#define SURF_SPEC            "Result/Retrieval_' + str(self.process) + '/surface_albedo.dat"\n',
                      '#define TIDELOCK             0\n',  # If the planet is tidally locked
                      '#define FaintSun             1.0\n',  # Faint early Sun factor
                      '#define STAR_TEMP            ' + str(self.param['Tirr']) + '\n',  # 394.109 irradiation Temperature at 1 AU
                      '#define THETAREF             1.0471\n',  # Slant Path Angle in radian
                      '#define PAB                  0.343\n',  # Planet Bond Albedo
                      '#define FADV                 0.25\n',  # Advection factor: 0.25=uniformly distributed, 0.6667=no Advection
                      # '#define PSURFAB              ' + str(float(self.param['Ag'])) + '\n',  # Planet Surface Albedo
                      '#define PSURFEM              0.0\n',  # Planet Surface Emissivity
                      '#define DELADJUST            1\n',  # Whether use the delta adjustment in the 2-stream diffuse radiation
                      '#define TAUTHRESHOLD         0.1\n',  # Optical Depth Threshold for multi-layer diffuse radiation
                      '#define TAUMAX               1000.0\n',  # Maximum optical Depth in the diffuse radiation
                      '#define TAUMAX1              1000.0\n',  # Maximum optical Depth in the diffuse radiation
                      '#define TAUMAX2              1000.0\n',
                      '#define IFDIFFUSE            1\n',  # Set to 1 if want to include diffuse solar radiation into the photolysis rate
                      #
                      '#define IFUVMULT             0\n',  # Whether do the UV Multiplying
                      '#define FUVMULT              1.0E+3\n',  # Multiplying factor for FUV radiation <200 nm
                      '#define MUVMULT              1.0E+2\n',  # Multiplying factor for MUV radiation 200 - 300 nm
                      '#define NUVMULT              1.0E+1\n',  # Multiplying factor for NUV radiation 300 - 400 nm
                      #
                      # Planet Temperature-Pressure Preofile
                      '#define TPMODE               1\n',  # 1: import data from a ZTP list
                      #                                      0: calculate TP profile from the parametized formula*/
                      '#define TPLIST               "Data/TP1986.dat"\n',
                      '#define PTOP                 1.0E-5\n',  # Pressure at the top of atmosphere in bar
                      '#define TTOP				    ' + str(self.param['Tp']) + '\n',  # Temperature at the top of atmosphere
                      '#define TSTR                 ' + str(self.param['Tp']) + '\n',  # Temperature at the top of stratosphere
                      '#define TINV                 0\n',  # set to 1 if there is a temperature inversion
                      '#define PSTR                 1.0E-1\n',  # Pressure at the top of stratosphere
                      '#define PMIDDLE				0\n',  # Pressure at the bottom of stratosphere
                      '#define TMIDDLE				' + str(self.param['Tp']) + '\n',  # Temperature at the bottom of stratosphere
                      '#define PBOTTOM				1.0E+0\n',  # Pressure at the bottom of stratosphere
                      '#define TBOTTOM				' + str(self.param['Tp']) + '\n',  # Temperature at the bottom of stratosphere
                      '#define PPOFFSET			    0.0\n',  # Pressure offset in log [Pa]
                      #
                      # Calculation Grids
                      # '#define zbin                 180\n',  # How many altitude bin?
                      '#define zbin                 ' + str(int(self.param['n_layer'])) + '\n',  # How many altitude bin?
                      # '#define zmax                 1631.0\n',  # Maximum altitude in km
                      # '#define zmin                 0.0\n',  # Maximum altitude in km
                      '#define WaveBin              9999\n',  # How many wavelength bin?
                      '#define WaveMin              1.0\n',  # Minimum Wavelength in nm
                      '#define WaveMax              10000.0\n',  # Maximum Wavelength in nm
                      '#define WaveMax1             1000.0\n',  # Maximum Wavelength in nm for the Calculation of UV-visible radiation and photolysis rates
                      '#define TDEPMAX	            300.0\n',  # Maximum Temperature-dependence Validity for UV Cross sections
                      '#define TDEPMIN              200.0\n',  # Minimum Temperature-dependence Validity for UV Cross sections

                      # The criteria of convergence
                      '#define Tol1                 1.0E+10\n',
                      '#define Tol2                 1.0E-16\n',
                      #
                      # Mode of iteration
                      '#define TSINI                1.0E-18\n',  # Initial Trial Timestep, generally 1.0E-8
                      '#define FINE1                1\n',  # Set to one for fine iteration: Set to 2 to disregard the bottom boundary layers
                      '#define FINE2                1\n',  # Set to one for fine iteration: Set to 2 to disregard the fastest varying point
                      '#define TMAX                 1.0E+12\n',  # Maximum of time step
                      '#define TMIN                 1.0E-25\n',  # Minimum of time step
                      '#define TSPEED               1.0E+12\n',  # Speed up factor
                      '#define NMAX                 1E+4\n',  # Maximum iteration cycles
                      '#define NMAXT                1.0E+13\n',  # Maximum iteration cumulative time in seconds
                      '#define MINNUM               1.0E-0\n',  # Minimum number density in denominator
                      #
                      # Molecular Species
                      '#define NSP                  111\n',  # Number of species in the standard list
                      '#define SPECIES_LIST         "Data/species_Earth_Full.dat"\n',
                      # '#define AIRM_FILE            "Result/Retrieval_' + str(self.process) + '/mean_mol_mass.dat"\n',
                      '#define AIRM                 ' + str(self.param['mean_mol_weight'][-1]) + '\n',  # Initial mean molecular mass of atmosphere, in atomic mass unit
                      '#define AIRVIS               1.0E-5\n',  # Dynamic viscosity in SI
                      # '#define RefIdxType           0\n',  # Type of Refractive Index: 0=Air, 1=CO2, 2=He, 3=N2, 4=NH3, 5=CH4, 6=H2, 7=O2, 8=composition
                      #
                      # Aerosol Species
                      '#define AERSIZE              1.0E-7\n',  # diameter in m
                      '#define AERDEN               1.84E+3\n',  # density in SI
                      '#define NCONDEN              1\n',  # Calculate the condensation every NCONDEN iterations
                      '#define IFGREYAER            0\n',  # Contribute to the grey atmosphere Temperature? 0=no, 1=yes
                      '#define SATURATIONREDUCTION  1.0\n',  # Ad hoc reduction factor for saturation pressure of water
                      '#define AERRADFILE1          "Data/H2SO4AER_CrossM_01.dat"\n',  # radiative properties of H2SO4
                      '#define AERRADFILE2          "Data/S8AER_CrossM_01.dat"\n',  # radiative properties of S8
                      #
                      # Initial Concentration Setting
                      '#define IMODE                4\n',  # 1: Import from SPECIES_LIST
                      #                                    # 0: Calculate initial concentrations from chemical equilibrium sub-routines (not rad)
                      #                                    # 3: Calculate initial concentrations from simplied chemical equilibrium formula (not rad)
                      #                                    # 2: Import from results of previous calculations
                      #                                    # 4: Import from results of previous calculations in the standard form (TP import only for rad)
                      '#define NATOMS               23\n',  # Number of atoms for chemical equil
                      '#define NMOLECULES           172\n',  # Number of molecules for chemical equil
                      '#define MOL_DATA_FILE        "Data/molecules_all.dat"\n',  # Data file for chemical equilibrium calculation
                      # '#define ATOM_ABUN_FILE       "Data/atom_H2O_CH4.dat"\n',  # Data file for chemical equilibrium calculation
                      '#define IMPORTFILEX          "Result/Aux/Conx.dat"\n',  # File of concentrations X to be imported
                      '#define IMPORTFILEF          "Result/Aux/Conf.dat"\n',  # File of concentrations F to be imported
                      # '#define IFIMPORTH2O          0\n',  # When H2O is set to constant, 1=import mixing ratios
                      # '#define IFIMPORTCO2          0\n',  # When CO is set to constant, 1=import mixing ratios
                      # Reaction Zones
                      '#define REACTION_LIST        "Data/zone_Earth_Full.dat"\n',
                      '#define NKin                 645\n',  # Number of Regular Chemical Reaction in the standard list
                      '#define NKinM                90\n',  # Number of Thermolecular Reaction in the standard list
                      '#define NKinT                93\n',  # Number of Thermal Dissociation Reaction in the standard list
                      '#define NPho                 71\n',  # Number of Photochemical Reaction in the standard list
                      '#define THREEBODY            1.0\n',  # Enhancement of THREEBODY Reaction when CO2 dominant
                      #
                      # Parametization of Eddy Diffusion Coefficient
                      '#define EDDYPARA             1\n',  # =1 from Parametization, =2 from imported list
                      '#define KET                  1.0E+6\n',  # unit cm2 s-1
                      '#define KEH                  1.0E+6\n',
                      '#define ZT                   200.0\n',  # unit km
                      '#define Tback                1E+4\n',
                      '#define KET1                 1.0E+6\n',
                      '#define KEH1                 1.0E+8\n',
                      '#define EDDYIMPORT           "Data/EddyH2.dat"\n',
                      '#define MDIFF_H_1            4.87\n',
                      '#define MDIFF_H_2            0.698\n',
                      '#define MDIFF_H2_1           2.80\n',
                      '#define MDIFF_H2_2           0.740\n',
                      '#define MDIFF_H2_F           1.0\n',
                      #
                      # Parameters of rainout rates
                      '#define RainF                0.0\n',  # Rainout factor, 0 for no rainout, 1 for earthlike normal rainout, <1 for reduced rainout
                      '#define CloudDen             1.0\n',  # Cloud density in the unit of g m-3
                      # Output Options
                      '#define OUT_DIR              "Result/Retrieval_' + str(self.process) + '/"\n',
                      '#define TINTSET              20.0\n',  # Internal Heat Temperature
                      '\n',
                      '#define OUT_STD              "Result/Jupiter_1/ConcentrationSTD.dat"\n',
                      '#define OUT_FILE1            "Result/GJ1214_Figure/Conx.dat"\n',
                      '#define OUT_FILE2            "Result/GJ1214_Figure/Conf.dat"\n',
                      '#define NPRINT               1E+2\n',  # Printout results and histories every NPRINT iterations
                      '#define HISTORYPRINT         0\n',  # print out time series of chemical composition if set to 1
                      #
                      # Input choices for the infrared opacities
                      # Must be set to the same as the opacity code
                      #
                      '#define CROSSHEADING         "Cross3/N2_FullT_LowRes/"\n',
                      #
                      '#define NTEMP                20\n',  # Number of temperature points in grid
                      '#define TLOW                 100.0\n',  # Temperature range in K
                      '#define THIGH                2000.0\n',
                      #
                      '#define NPRESSURE            10\n',  # Number of pressure points in grid
                      '#define PLOW                 1.0e-01\n',  # Pressure range in Pa
                      '#define PHIGH                1.0e+08\n',
                      #
                      '#define NLAMBDA              16000\n',  # Number of wavelength points in grid
                      '#define LAMBDALOW            1.0e-07\n',  # Wavelength range in m -> 0.1 micron
                      '#define LAMBDAHIGH           2.0e-04\n',  # in m -> 200 micron
                      '#define LAMBDATYPE           1\n',  # LAMBDATYPE=1 -> constant resolution
                      #                                    # LAMBDATYPE=2 -> constant wave step
                      #
                      #
                      # IR emission spectra output options
                      '#define IRLamMin             1.0\n',  # Minimum wavelength in the IR emission output, in microns
                      '#define IRLamMax             100.0\n',  # Maximum wavelength in the IR emission output, in microns, was 100
                      '#define IRLamBin             9999\n',  # Number of wavelength bin in the IR emission spectra, was 9999
                      '#define Var1STD              7\n',
                      '#define Var2STD              20\n',
                      '#define Var3STD              21\n',
                      '#define Var4STD              52\n',
                      '#define Var1RATIO            0.0\n',
                      '#define Var2RATIO            0.0\n',
                      '#define Var3RATIO            0.0\n',
                      '#define Var4RATIO            0.0\n',
                      #
                      #  Stellar Light Reflection output options
                      '#define UVRFILE              "Result/Jupiter_1/Reflection"\n',  # Output spectrum file name
                      '#define UVRFILEVar1          "Result/Jupiter_1/ReflectionVar1.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar2          "Result/Jupiter_1/ReflectionVar2.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar3          "Result/Jupiter_1/ReflectionVar3.dat"\n',  # Output spectrum file name
                      '#define UVRFILEVar4          "Result/Jupiter_1/ReflectionVar4.dat"\n',  # Output spectrum file name
                      '#define UVROPTFILE           "Result/Jupiter_1/UVROpt.dat"\n',  # Output spectrum file name
                      '#define AGFILE               "Result/Jupiter_1/GeometricA.dat"\n',  # Output spectrum file name
                      #
                      # Stellar Light Transmission output options
                      '#define UVTFILE              "Result/Jupiter_1/Transmission.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar1          "Result/Jupiter_1/TransmissionVar1.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar2          "Result/Jupiter_1/TransmissionVar2.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar3          "Result/Jupiter_1/TransmissionVar3.dat"\n',  # Output spectrum file name
                      '#define UVTFILEVar4          "Result/Jupiter_1/TransmissionVar4.dat"\n',  # Output spectrum file name
                      '#define UVTOPTFILE           "Result/Jupiter_1/UVTOpt.dat"\n',  # Output spectrum file name
                      #
                      # Thermal Emission output options
                      '#define IRFILE               "Result/Jupiter_1/Emission.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar1           "Result/Jupiter_1/EmissionVar1.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar2           "Result/Jupiter_1/EmissionVar2.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar3           "Result/Jupiter_1/EmissionVar3.dat"\n',  # Output spectrum file name
                      '#define IRFILEVar4           "Result/Jupiter_1/EmissionVar4.dat"\n',  # Output spectrum file name
                      '#define IRCLOUDFILE          "Result/Jupiter_1/CloudTopE.dat"\n',  # Output emission cloud top file name
                      #
                      # Cloud Top Determination
                      '#define OptCloudTop          1.0\n',  # Optical Depth of the Cloud Top
                      #
                      '#endif\n',
                      #
                      # 1 Tg yr-1 = 3.7257E+9 H /cm2/s for earth
                      ]
        with open(self.c_code_directory + 'par_' + str(self.process) + '.h', 'w') as file:
            for riga in c_par_file:
                file.write(riga)

    def __core_c_file(self):
        # if self.param['spectrum']['wl'][0] <= 0.4 and 0.75 < self.param['spectrum']['wl'][-1] < 1.1:
        #     iniz, fine = 1350, 4900
        # elif self.param['spectrum']['wl'][0] >= 0.4 and 0.75 < self.param['spectrum']['wl'][-1] < 1.1:
        #     iniz, fine = 3000, 4900
        # elif 0.4 <= self.param['spectrum']['wl'][0] < 0.9 and self.param['spectrum']['wl'][-1] > 1.1:
        #     iniz, fine = 3000, 6150
        # elif self.param['spectrum']['wl'][0] >= 0.9:
        #     iniz, fine = 4800, 6150
        # else:
        #     iniz, fine = 1350, 6150
        iniz = self.param['start_c_wl_grid'] + 0.0
        fine = self.param['stop_c_wl_grid'] + 0.0

        c_core_file = ['#include <stdio.h>\n',
                       '#include <math.h>\n',
                       '#include <stdlib.h>\n',
                       '#include <string.h>\n',

                       '#include "par_' + str(self.process) + '.h"\n',

                       '#include "constant.h"\n',
                       '#include "routine.h"\n',
                       '#include "global_rad_gasplanet.h"\n',
                       '#include "GetData.c"\n',
                       '#include "Interpolation.c"\n',
                       '#include "nrutil.h"\n',
                       '#include "nrutil.c"\n',
                       '#include "Convert.c"\n',
                       '#include "TPPara.c"\n',
                       # #include "TPScale.c"\n',
                       '#include "RefIdx.c"\n',
                       '#include "readcross.c"\n',
                       '#include "readcia.c"\n',
                       '#include "Reflection_.c"\n',
                       '#include "Trapz.c"\n',

                       # external (global) variables

                       'double thickl[zbin];\n',
                       'double zl[zbin+1];\n',
                       'double pl[zbin+1];\n',
                       'double tl[zbin+1];\n',
                       'double MM[zbin+1];\n',
                       'double MMZ[zbin+1];\n',
                       'double wavelength[NLAMBDA];\n',
                       'double solar[NLAMBDA];\n',
                       'double PSURFAB[NLAMBDA];\n',
                       'double crossr[zbin+1][NLAMBDA], crossr_H2O[NLAMBDA], crossr_CH4[NLAMBDA], crossr_CO2[NLAMBDA], crossr_CO[NLAMBDA], crossr_O2[NLAMBDA], crossr_O3[NLAMBDA], crossr_N2O[NLAMBDA], crossr_N2[NLAMBDA], crossr_H2[NLAMBDA];\n',
                       'double crossa[3][NLAMBDA], sinab[3][NLAMBDA], asym[3][NLAMBDA];\n',
                       'double **opacH2O, **opacNH3, **opacCH4, **opacH2S, **opacSO2, **opacCO2, **opacCO, **opacO2, **opacO3, **opacN2O, **opacN2;\n',

                       #double **opacH2O2, **opacHO2; \n',
                       #double **opacC2H2, **opacC2H4, **opacC2H6, **opacHCN, **opacCH2O2, **opacHNO3;\n',
                       #double **opacNO, **opacNO2, **opacOCS;\n',
                       #double **opacHF, **opacHCl, **opacHBr, **opacHI, **opacClO, **opacHClO;\n',
                       #double **opacHBrO, **opacPH3, **opacCH3Cl, **opacCH3Br, **opacDMS, **opacCS2;\n',

                       'int    ReactionR[NKin+1][7], ReactionM[NKinM+1][5], ReactionP[NPho+1][9], ReactionT[NKinT+1][4];\n',
                       'int    numr=0, numm=0, numt=0, nump=0, numx=0, numc=0, numf=0, numa=0, waternum=0, waterx=0;\n',
                       'double **xx, **xx1, **xx2, **xx3, **xx4;\n',
                       'double TransOptD[zbin+1][NLAMBDA], RefOptD[zbin+1][NLAMBDA];\n',
                       # /*double H2CIA[zbin+1][NLAMBDA], H2HeCIA[zbin+1][NLAMBDA], N2CIA[zbin+1][NLAMBDA], CO2CIA[zbin+1][NLAMBDA];*/\n',
                       'double H2H2CIA[zbin+1][NLAMBDA], H2HeCIA[zbin+1][NLAMBDA], H2HCIA[zbin+1][NLAMBDA], N2H2CIA[zbin+1][NLAMBDA], N2N2CIA[zbin+1][NLAMBDA], CO2CO2CIA[zbin+1][NLAMBDA], O2O2CIA[zbin+1][NLAMBDA];\n',
                       'double cH2O[zbin+1][NLAMBDA], aH2O[zbin+1][NLAMBDA], gH2O[zbin+1][NLAMBDA];\n',
                       'double cNH3[zbin+1][NLAMBDA], aNH3[zbin+1][NLAMBDA], gNH3[zbin+1][NLAMBDA];\n',

                       'int main()\n',
                       '{\n',
                       '    int s,i,ii,j,jj,jjj,k,nn,qytype,stdnum;\n',
                       '    int nums, numx1=1, numf1=1, numc1=1, numr1=1, numm1=1, nump1=1, numt1=1;\n',
                       '    char *temp;\n',
                       '    char dataline[10000];\n',
                       '    double temp1, wavetemp, crosstemp, DD, GA, mixtemp;\n',
                       '    double z[zbin+1], T[zbin+1], PP[zbin+1], P[zbin+1];\n',
                       '    double *wavep, *crossp, *crosspa, *qyp, *qyp1, *qyp2, *qyp3, *qyp4, *qyp5, *qyp6, *qyp7, **cross, **qy;\n',
                       '    double **crosst, **qyt, *crosspt, *qypt, *qyp1t, *qyp2t, *qyp3t, *qyp4t, *qyp5t, *qyp6t, *qyp7t;\n',
                       '    FILE *fspecies, *fzone, *fhenry, *fp, *fp1, *fp2, *fp3;\n',
                       '    FILE *fout, *fout1, *fout3, *fout4, *fcheck, *ftemp, *fout5, *foutp, *foutc;\n',
                       '    FILE *fimport, *fimportcheck;\n',
                       '    FILE *TPPrint;\n',

                       '    xx = dmatrix(1,zbin,1,NSP);\n',
                       '    xx1 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx2 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx3 = dmatrix(1,zbin,1,NSP);\n',
                       '    xx4 = dmatrix(1,zbin,1,NSP);\n',

                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=1; i<=NSP; i++) {\n',
                       '            xx[j][i] = 0.0;\n',
                       '            xx1[j][i] = 0.0;\n',
                       '            xx2[j][i] = 0.0;\n',
                       '            xx3[j][i] = 0.0;\n',
                       '            xx4[j][i] = 0.0;\n',
                       '        }\n',
                       '    }\n',

                       #    GA = GRAVITY*MASS_PLANET/RADIUS_PLANET/RADIUS_PLANET;\n',  # Planet Surface Gravity Acceleration, in SI

                       #    Set the wavelength for calculation
                       '    double dlambda, start, interval, lam[NLAMBDA];\n',
                       '    start = log10(LAMBDALOW);\n',
                       '    interval = log10(LAMBDAHIGH) - log10(LAMBDALOW);\n',
                       '    dlambda = interval / (NLAMBDA-1.0);\n',
                       '    for (i=0; i<NLAMBDA; i++){\n',
                       '        wavelength[i] = pow(10.0, start+i*dlambda)*1.0E+9;\n',  # in nm
                       '        lam[i] = wavelength[i]*1.0E-3;\n',  # in microns
                       '    }\n',

                       # Obtain the stellar radiation
                       '    fp2 = fopen(STAR_SPEC,"r");\n',
                       '    fp3 = fopen(STAR_SPEC,"r");\n',
                       '    s = LineNumber(fp2, 1000);\n',
                       '    double swave[s], sflux[s];\n',
                       '    GetData(fp3, 1000, s, swave, sflux);\n',
                       '    fclose(fp2);\n',
                       '    fclose(fp3);\n',
                       '    Interpolation(wavelength, NLAMBDA, solar, swave, sflux, s, 0);\n',
                       '    for (i=0; i<NLAMBDA; i++) {\n',
                       '        solar[i] = solar[i]/ORBIT/ORBIT*FaintSun;\n',  # convert from flux at 1 AU
                       '    }\n',
                       '    i=0;\n',
                       '    while (solar[i]>0 || wavelength[i]<9990 ) { i++;}\n',
                       '    for (j=i; j<NLAMBDA; j++) {\n',
                       '        solar[j] = solar[i-1]*pow(wavelength[i-1],4)/pow(wavelength[j],4);\n',
                       '    }\n',
                       # '\t  printf("%s\\n", "The stellar radiation data are imported.");\n',
                       '    fp2 = fopen(SURF_SPEC,"r");\n',
                       '    char dataline2[100];\n',
                       '    i=0;\n',
                       '    while (i < NLAMBDA && fgets(dataline2, sizeof(dataline2), fp2) != NULL) {\n',
                       '        PSURFAB[i] = atof(dataline2);\n',  # convert string to double and store in array
                       '        i++;\n',
                       '    }\n',
                       '    fclose(fp2);\n',
                       # Import Species List
                       '    fspecies=fopen(SPECIES_LIST, "r");\n',
                       '    s=LineNumber(fspecies, 10000);\n',
                       # '\tprintf("Species list: \\n");\n',
                       '    fclose(fspecies);\n',
                       '    fspecies=fopen(SPECIES_LIST, "r");\n',
                       '    struct Molecule species[s];\n',
                       '    temp=fgets(dataline, 10000, fspecies);\n',  # Read in the header line
                       '    i=0;\n',
                       '    while (fgets(dataline, 10000, fspecies) != NULL )\n',
                       '    {\n',
                       '        sscanf(dataline, "%s %s %d %d %lf %lf %d %lf %lf", (species+i)->name, (species+i)->type, &((species+i)->num), &((species+i)->mass), &((species+i)->mix), &((species+i)->upper), &((species+i)->lowertype), &((species+i)->lower), &((species+i)->lower1));\n',
                       # '\t\tprintf("%s %s %d %d %lf %lf %d %lf %lf\\n",(species+i)->name, (species+i)->type, (species+i)->num, (species+i)->mass, (species+i)->mix, (species+i)->upper, (species+i)->lowertype, (species+i)->lower, (species+i)->lower1);\n',
                       '        if (strcmp("X",species[i].type)==0) {numx=numx+1;}\n',
                       '        if (strcmp("F",species[i].type)==0) {numf=numf+1;}\n',
                       '        if (strcmp("C",species[i].type)==0) {numc=numc+1;}\n',
                       '        if (strcmp("A",species[i].type)==0) {numx=numx+1; numa=numa+1;}\n',
                       '        i=i+1;\n',
                       '    }\n',
                       '    fclose(fspecies);\n',
                       '    nums=numx+numf+numc;\n',
                       # '\tprintf("%s\\n", "The species list is imported.");\n',
                       # '\tprintf("%s %d\\n", "Number of species in model:", nums);\n',
                       # '\tprintf("%s %d\\n", "Number of species to be solved in full:", numx);\n',
                       # '\tprintf("%s %d\\n", "In which the number of aerosol species is:", numa);\n',
                       # '\tprintf("%s %d\\n", "Number of species to be solved in photochemical equil:", numf);\n',
                       # '\tprintf("%s %d\\n", "Number of species assumed to be constant:", numc);\n',
                       '    int labelx[numx+1], labelc[numc+1], labelf[numf+1], MoleculeM[numx+1], listAER[numa+1], AERCount=1;\n',
                       '    for (i=0; i<s; i++) {\t\t\t\n',
                       '        if (strcmp("X",species[i].type)==0 || strcmp("A",species[i].type)==0) {\n',
                       '            labelx[numx1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) { \n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       '            if (species[i].num==7) {\n',
                       '                waternum=numx1;\n',
                       '                waterx=1;\n',
                       '            }\n',
                       '            MoleculeM[numx1]=species[i].mass;\n',
                       '            if (species[i].lowertype==1) {\n',
                       '                xx[1][species[i].num]=species[i].lower1*MM[1];\n',
                       '            }\n',
                       '            if (strcmp("A",species[i].type)==0) {\n',
                       '                listAER[AERCount]=numx1;\n',
                       '                AERCount = AERCount+1;\n',
                       #                printf("%s %d\\n", "The aerosol species is", numx1);\n',
                       '            }\n',
                       '            numx1=numx1+1;\n',
                       '        }\n',
                       '        if (strcmp("F",species[i].type)==0) {\n',
                       '            labelf[numf1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) { \n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       '            numf1=numf1+1;\n',
                       '        }\n',
                       '        if (strcmp("C",species[i].type)==0) {\n',
                       '            labelc[numc1]=species[i].num;\n',
                       '            for (j=1; j<=zbin; j++) {\n',
                       '                xx[j][species[i].num]=MM[j]*species[i].mix;\n',
                       '            }\n',
                       # import constant mixing ratio list for H2O
                       #            if (IFIMPORTH2O == 1 && species[i].num == 7) {\n',
                       #                fimport=fopen("Data/ConstantMixing.dat", "r");\n',
                       #                fimportcheck=fopen("Data/ConstantMixingH2O.dat", "w");\n',
                       #                temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       #                for (j=1; j<=zbin; j++) {\n',
                       #                    fscanf(fimport, "%lf\\t", &temp1);\n',
                       #                    fscanf(fimport, "%le\\t", &mixtemp);\n',
                       #                    fscanf(fimport, "%le\\t", &temp1);\n',
                       #                    xx[j][7]=mixtemp * MM[j];\n',
                       #                    fprintf(fimportcheck, "%f\\t%e\\t%e\\n", zl[j], mixtemp, xx[j][7]);\n',
                       #                }\n',
                       #                fclose(fimport);\n',
                       #                fclose(fimportcheck);\n',
                       #            }\n',
                       # import constant mixing ratio list for CO2
                       #            if (IFIMPORTCO2 == 1 && species[i].num == 52) {\n',
                       #                fimport=fopen("Data/ConstantMixing.dat", "r");\n',
                       #                fimportcheck=fopen("Data/ConstantMixingCO2.dat", "w");\n',
                       #                temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       #                for (j=1; j<=zbin; j++) {\n',
                       #                    fscanf(fimport, "%lf\\t", &temp1);\n',
                       #                    fscanf(fimport, "%le\\t", &temp1);\n',
                       #                    fscanf(fimport, "%le\\t", &mixtemp);\n',
                       #                    xx[j][52]=mixtemp * MM[j];\n',
                       #                    fprintf(fimportcheck, "%f\\t%e\\t%e\\n", zl[j], mixtemp, xx[j][52]);\n',
                       #                }\n',
                       #                fclose(fimport);\n',
                       #                fclose(fimportcheck);\n',
                       #            }\n',
                       '            numc1=numc1+1;\n',
                       '        }\n',
                       '    }\n',
                       '    fimport=fopen(IMPORTFILEX, "r");\n',
                       #    fimportcheck=fopen("Data/Fimportcheck.dat","w");\n',
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',
                       #        fprintf(fimportcheck, "%lf\\t", temp1);\n',
                       '        for (i=1; i<=numx; i++) {\n',
                       '            fscanf(fimport, "%le\\t\\t", &xx[j][labelx[i]]);\n',
                       #            fprintf(fimportcheck, "%e\\t", xx[j][labelx[i]]);\n',
                       '        }\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',  # column of air
                       #        fprintf(fimportcheck,"\\n");\n',
                       '    }\n',
                       '    fclose(fimport);\n',
                       '    fimport=fopen(IMPORTFILEF, "r");\n',
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',
                       #        fprintf(fimportcheck, "%lf\\t", temp1);\n',
                       '        for (i=1; i<=numf; i++) {\n',
                       '            fscanf(fimport, "%le\\t\\t", &xx[j][labelf[i]]);\n',
                       #            fprintf(fimportcheck, "%e\\t", xx[j][labelf[i]]);\n',
                       '        }\n',
                       '        fscanf(fimport, "%lf\\t\\t", &temp1);\n',  # column of air
                       #        fprintf(fimportcheck,"\\n");\n',
                       '    }\n',
                       '    fclose(fimport);\n',
                       #    fclose(fimportcheck);\n',

                       # Set up atmospheric profiles

                       '    char outstd[1024];\n',
                       '    strcpy(outstd,OUT_DIR);\n',
                       '    strcat(outstd,"ConcentrationSTD.dat");\n',
                       '    if (IMODE == 4) {\n',  # Import the computed profile directly
                       '        fimport=fopen(outstd, "r");\n',
                       #        fimportcheck=fopen("Data/Fimportcheck.dat","w");\n',
                       '        temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '        temp=fgets(dataline, 10000, fimport);\n',  # Read in the header line
                       '        for (j=1; j<=zbin; j++) {\n',
                       '            fscanf(fimport, "%lf\\t", &zl[j]);\n',
                       #            fprintf(fimportcheck, "%lf\\t", zl[j]);\n',
                       '            fscanf(fimport, "%lf\\t", &z[j-1]);\n',
                       '            fscanf(fimport, "%lf\\t", &z[j]);\n',
                       '            fscanf(fimport, "%lf\\t", &tl[j]);\n',
                       '            fscanf(fimport, "%le\\t", &pl[j]);\n',
                       '            MM[j] = pl[j]/KBOLTZMANN/tl[j]*1.0E-6;\n',
                       '            for (i=1; i<=NSP; i++) {\n',
                       '                fscanf(fimport, "%le\\t", &xx[j][i]);\n',
                       #                fprintf(fimportcheck, "%e\\t", xx[j][i]);\n',
                       #                MM[j] += xx[j][i];\n',
                       '            }\n',
                       #            printf("%s %f %f\\n", "TP", tl[j], pl[j]);\n',
                       #            fprintf(fimportcheck,"\\n");\n',
                       '        }\n',
                       '        fclose(fimport);\n',
                       #        fclose(fimportcheck);\n',
                       # '        thickl = (z[zbin]-z[zbin-1])*1.0E+5;\n',
                       '        for (j=1; j<=zbin; j++) {\n',
                       '            thickl[j] = (z[j]-z[j-1])*1.0E+5;\n',
                       # '            printf("%f\\n", thickl[j]);\n',
                       '        }\n',
                       '        for (j=1; j<zbin; j++) {\n',
                       '            T[j] = (tl[j] + tl[j+1])/2.0;\n',
                       '        }\n',
                       '        T[0] = 1.5*tl[1] - 0.5*tl[2];\n',
                       '        T[zbin] = 1.5*tl[zbin] - 0.5*tl[zbin-1];\n',
                       '    }\n',

                       #    Rayleigh Scattering
                       '    double refidx0,DenS;\n',
                       '    DenS = 101325.0 / KBOLTZMANN / 273.0 * 1.0E-6;\n',
                       # '    for (i=0; i<NLAMBDA; i++){\n',
                       '    for (i=' + str(iniz) + '; i<' + str(fine) + '; i++){\n',
                       #        if (RefIdxType == 0) { refidx0=AirRefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 1) { refidx0=CO2RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 2) { refidx0=HeRefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 3) { refidx0=N2RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 4) { refidx0=NH3RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 5) { refidx0=CH4RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 6) { refidx0=H2RefIdx(wavelength[i]);}\n',
                       #        if (RefIdxType == 7) { refidx0=O2RefIdx(wavelength[i]);}\n',
                       #        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       #        crossr[i]=1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       #        if (RefIdxType == 6) {crossr[i] = 8.14e-13*pow(wavelength[i]*10.0,-4)+1.28e-6*pow(wavelength[i]*10.0,-6)+1.61*pow(wavelength[i]*10.0,-8); }\n',  # Dalgarno 1962
                       #        if (RefIdxType == 8) {\n',
                       '        refidx0 = H2ORefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_H2O[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = CH4RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_CH4[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = CO2RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_CO2[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = CORefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_CO[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = O2RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_O2[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = O3RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_O3[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = N2ORefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_N2O[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        refidx0 = N2RefIdx(wavelength[i]);\n',
                       '        if (refidx0 < 1.0) { refidx0 = 1.0; }\n',
                       '        crossr_N2[i] = 1.061*8.0*pow(PI,3)*pow(pow(refidx0,2)-1,2)/3.0/pow(wavelength[i]*1.0E-7,4)/DenS/DenS;\n',
                       '        crossr_H2[i] = 8.14e-13*pow(wavelength[i]*10.0,-4)+1.28e-6*pow(wavelength[i]*10.0,-6)+1.61*pow(wavelength[i]*10.0,-8);\n',  # Dalgarno 1962
                       '        for (j=1; j<=zbin; j++) {\n']
        if not self.param['contribution']:
            c_core_file += ['            crossr[j][i] = ((crossr_H2O[i] * (xx[j][7]/MM[j])) + (crossr_CH4[i] * (xx[j][21]/MM[j])) + (crossr_CO2[i] * (xx[j][52]/MM[j])) + (crossr_CO[i] * (xx[j][20]/MM[j])) + (crossr_O2[i] * (xx[j][54]/MM[j])) + (crossr_O3[i] * (xx[j][2]/MM[j])) + (crossr_N2O[i] * (xx[j][11]/MM[j])) + (crossr_N2[i] * (xx[j][55]/MM[j])) + (crossr_H2[i] * (xx[j][53]/MM[j]))) / ((xx[j][7]/MM[j]) + (xx[j][21]/MM[j]) + (xx[j][52]/MM[j]) + (xx[j][20]/MM[j]) + (xx[j][54]/MM[j]) + (xx[j][2]/MM[j]) + (xx[j][11]/MM[j]) + (xx[j][55]/MM[j]) + (xx[j][53]/MM[j]));\n']
        elif self.param['contribution'] and self.param['mol_contr'] is not None:
            c_core_file += ['            crossr[j][i] = ((crossr_H2O[i] * (xx[j][7]/MM[j])) + (crossr_CH4[i] * (xx[j][21]/MM[j])) + (crossr_CO2[i] * (xx[j][52]/MM[j])) + (crossr_CO[i] * (xx[j][20]/MM[j])) + (crossr_O2[i] * (xx[j][54]/MM[j])) + (crossr_O3[i] * (xx[j][2]/MM[j])) + (crossr_N2O[i] * (xx[j][11]/MM[j])) + (crossr_N2[i] * (xx[j][55]/MM[j])) + (crossr_H2[i] * (xx[j][53]/MM[j])));\n']
        else:
            c_core_file += ['            crossr[j][i] = 0.0;\n']
        c_core_file+= ['        }\n',
                       '    }\n',

                       '    readcia();\n',

                       # check CIA
                       #    for (i=0; i<NLAMBDA; i++) {\n',
                       #    printf("%s\\t%f\\t%e\\t%e\\t%e\\t%e\\n", "CIA", wavelength[i], H2CIA[1][i], H2HeCIA[1][i], N2CIA[1][i], CO2CIA[1][i]);\n',
                       #    }\n',
                       #
                       # '\tprintf("%s\\n", "Collision-induced absorption cross sections are imported ");\n',

                       # Obtain the opacity
                       '    opacH2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacNH3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCH4 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacH2S = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacSO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacCO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacO3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacN2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       '    opacN2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacOH = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacH2CO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacH2O2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacC2H2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacC2H4 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacC2H6 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHCN = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCH2O2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHNO3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacN2O = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacNO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacNO2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacOCS = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHF = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHCl = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHBr = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHI = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacClO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHClO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacHBrO = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacPH3 = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCH3Cl = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCH3Br = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacDMS = dmatrix(1,zbin,0,NLAMBDA-1);\n',
                       #    opacCS2 = dmatrix(1,zbin,0,NLAMBDA-1);\n',

                       '    char crossfile[1024];\n']

        for mol in self.param['fit_molecules']:
            if mol != 'H2':
                c_core_file += ['    strcpy(crossfile,CROSSHEADING);\n',
                                '    strcat(crossfile,"opac' + mol + '.dat");\n',
                                '    readcross(crossfile, opac' + mol + ');\n']

                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCO.dat");\n',
                       #    readcross(crossfile, opacCO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacN2O.dat");\n',
                       #    readcross(crossfile, opacN2O);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacOH.dat");\n',
                       #    readcross(crossfile, opacOH);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2CO.dat");\n',
                       #    readcross(crossfile, opacH2CO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacH2O2.dat");\n',
                       #    readcross(crossfile, opacH2O2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHO2.dat");\n',
                       #    readcross(crossfile, opacHO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H2.dat");\n',
                       #    readcross(crossfile, opacC2H2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H4.dat");\n',
                       #    readcross(crossfile, opacC2H4);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacC2H6.dat");\n',
                       #    readcross(crossfile, opacC2H6);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHCN.dat");\n',
                       #    readcross(crossfile, opacHCN);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacCH2O2.dat");\n',
                       #    readcross(crossfile, opacCH2O2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacHNO3.dat");\n',
                       #    readcross(crossfile, opacHNO3);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacN2.dat");\n',
                       #    readcross(crossfile, opacN2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacNO.dat");\n',
                       #    readcross(crossfile, opacNO);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacNO2.dat");\n',
                       #    readcross(crossfile, opacNO2);\n',
                       #
                       #    strcpy(crossfile,CROSSHEADING);\n',
                       #    strcat(crossfile,"opacOCS.dat");\n',
                       #    readcross(crossfile, opacOCS);\n',
                       #
                       #    foutc = fopen("Data/IRCross.dat","w");\n',
                       #    for (i=0; i<NLAMBDA; i++) {\n',
                       #        fprintf(foutc, "%f\\t", wavelength[i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2O[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCH4[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNH3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacO3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacN2O[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacSO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacOH[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2CO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2O2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacH2S[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H4[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacC2H6[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHCN[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacCH2O2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacHNO3[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacN2[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNO[1][i]);\n',
                       #        fprintf(foutc, "%e\\t", opacNO2[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacOCS[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHF[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHCl[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHBr[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHI[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacClO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHClO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacHBrO[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacPH3[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCH3Cl[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCH3Br[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacDMS[1][i]);\n',
                       #        fprintf(foutc, "%e\\n", opacCS2[1][i]);*/\n',
                       #    }\n',
                       #    fclose(foutc);\n',
                       #
                       #    tprintf("%s\\n", "Molecular cross sections are imported ");\n',

                       # Get Reaction List
        c_core_file+= ['    fzone=fopen(REACTION_LIST, "r");\n',
                       '    s=LineNumber(fzone, 10000);\n',
                       '    fclose(fzone);\n',
                       '    fzone=fopen(REACTION_LIST, "r");\n',
                       '    struct Reaction React[s];\n',
                       '    temp=fgets(dataline, 10000, fzone);\n',  # Read in the header line
                       '    i=0;\n',
                       '    while (fgets(dataline, 10000, fzone) != NULL )\n',
                       '    {\n',
                       '        sscanf(dataline, "%d %s %d", &((React+i)->dum), (React+i)->type, &((React+i)->num));\n',
                       #        printf("%d %s %d\\n", (React+i)->dum, React[i].type, React[i].num);\n',
                       '        if (strcmp("R",React[i].type)==0) {numr=numr+1;}\n',
                       '        if (strcmp("M",React[i].type)==0) {numm=numm+1;}\n',
                       '        if (strcmp("P",React[i].type)==0) {nump=nump+1;}\n',
                       '        if (strcmp("T",React[i].type)==0) {numt=numt+1;}\n',
                       '        i=i+1;\n',
                       '    }\n',
                       '    fclose(fzone);\n',
                       '    int zone_r[numr+1], zone_m[numm+1], zone_p[nump+1], zone_t[numt+1];\n',
                       '    for (i=0; i<s; i++) {\n',
                       '        if (strcmp("R",React[i].type)==0) {\n',
                       '            zone_r[numr1]=React[i].num;\n',
                       '            numr1=numr1+1;\n',
                       '        }\n',
                       '        if (strcmp("M",React[i].type)==0) {\n',
                       '            zone_m[numm1]=React[i].num;\n',
                       '            numm1=numm1+1;\n',
                       '        }\n',
                       '        if (strcmp("P",React[i].type)==0) {\n',
                       '            zone_p[nump1]=React[i].num;\n',
                       '            nump1=nump1+1;\n',
                       '        }\n',
                       '        if (strcmp("T",React[i].type)==0) {\n',
                       '            zone_t[numt1]=React[i].num;\n',
                       '        numt1=numt1+1;\n',
                       '        }\n',
                       '    }\n',
                       #    printf("%s\\n", "The reaction lists are imported.");\n',
                       #    tprintf("%s %d\\n", "Number of bi-molecular reactions:", numr);\n',
                       #    tprintf("%s %d\\n", "Number of tri-molecular reactions:", numm);\n',
                       #    tprintf("%s %d\\n", "Number of photolysis:", nump);\n',
                       #    tprintf("%s %d\\n", "Number of thermo-dissociations:", numt);\n',
                       '    GetReaction();\n',

                       # get the cross sections and quantum yields of molecules
                       '    cross=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    crosst=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    qy=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    qyt=dmatrix(1,nump,0,NLAMBDA-1);\n',
                       '    int stdcross[nump+1];\n',
                       '    double qysum[nump+1];\n',
                       #    fcheck=fopen("Data/CrossSectionCheck.dat","w"); \n',
                       '    for (i=1; i<=nump; i++) {\n',
                       '        stdcross[i]=ReactionP[zone_p[i]][1];\n',
                       '        qytype=ReactionP[zone_p[i]][8];\n',
                       '        qysum[i]=ReactionP[zone_p[i]][7];\n',
                       '        j=0;\n',
                       '        while (species[j].num != stdcross[i]) {j=j+1;}\n',
                       #        printf("%s\\n",species[j].name);\n',
                       '        fp=fopen(species[j].name, "r");\n',
                       '        fp1=fopen(species[j].name, "r");\n',
                       '        s=LineNumber(fp, 1000);\n',
                       #        printf("%d\\n",s);\n',
                       '        wavep=dvector(0,s-1);\n',
                       '        crossp=dvector(0,s-1);\n',
                       '        qyp=dvector(0,s-1);\n',
                       '        qyp1=dvector(0,s-1);\n',
                       '        qyp2=dvector(0,s-1);\n',
                       '        qyp3=dvector(0,s-1);\n',
                       '        qyp4=dvector(0,s-1);\n',
                       '        qyp5=dvector(0,s-1);\n',
                       '        qyp6=dvector(0,s-1);\n',
                       '        qyp7=dvector(0,s-1);\n',
                       '        crosspt=dvector(0,s-1);\n',
                       '        qypt=dvector(0,s-1);\n',
                       '        qyp1t=dvector(0,s-1);\n',
                       '        qyp2t=dvector(0,s-1);\n',
                       '        qyp3t=dvector(0,s-1);\n',
                       '        qyp4t=dvector(0,s-1);\n',
                       '        qyp5t=dvector(0,s-1);\n',
                       '        qyp6t=dvector(0,s-1);\n',
                       '        qyp7t=dvector(0,s-1);\n',
                       '        k=0;\n',
                       '        if (qytype==1) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf", wavep+k, crossp+k, crosspt+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==2) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==3) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==4) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==5) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==6) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==7) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp6+k, qyp6t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        if (qytype==8) {\n',
                       '            while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '                sscanf(dataline, "%lf %le %le %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", wavep+k, crossp+k, crosspt+k, qyp1+k, qyp1t+k, qyp2+k, qyp2t+k, qyp3+k, qyp3t+k, qyp4+k, qyp4t+k, qyp5+k, qyp5t+k, qyp6+k, qyp6t+k, qyp7+k, qyp7t+k, qyp+k, qypt+k);\n',
                       '                k=k+1; }\n',
                       '        }\n',
                       '        fclose(fp);\n',
                       '        fclose(fp1);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(cross+i), wavep, crossp, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(qy+i), wavep, qyp, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(crosst+i), wavep, crosspt, s, 0);\n',
                       '        Interpolation(wavelength, NLAMBDA, *(qyt+i), wavep, qypt, s, 0);\n',
                       '        free_dvector(wavep,0,s-1);\n',
                       '        free_dvector(crossp,0,s-1);\n',
                       '        free_dvector(qyp,0,s-1);\n',
                       '        free_dvector(qyp1,0,s-1);\n',
                       '        free_dvector(qyp2,0,s-1);\n',
                       '        free_dvector(qyp3,0,s-1);\n',
                       '        free_dvector(qyp4,0,s-1);\n',
                       '        free_dvector(qyp5,0,s-1);\n',
                       '        free_dvector(qyp6,0,s-1);\n',
                       '        free_dvector(qyp7,0,s-1);\n',
                       '        free_dvector(crosspt,0,s-1);\n',
                       '        free_dvector(qypt,0,s-1);\n',
                       '        free_dvector(qyp1t,0,s-1);\n',
                       '        free_dvector(qyp2t,0,s-1);\n',
                       '        free_dvector(qyp3t,0,s-1);\n',
                       '        free_dvector(qyp4t,0,s-1);\n',
                       '        free_dvector(qyp5t,0,s-1);\n',
                       '        free_dvector(qyp6t,0,s-1);\n',
                       '        free_dvector(qyp7t,0,s-1);\n',
                       #        printf("%s %s %s\\n", "The", species[j].name, "Cross section and quantum yield data are imported.");\n',
                       #        fprintf(fcheck, "%s %s %s\\n", "The", species[j].name, "Cross section and quantum yield data are imported.");\n',
                       #        for (j=0; j<NLAMBDA;j++) {fprintf(fcheck, "%lf %le %le %lf %lf\\n", wavelength[j], cross[i][j], crosst[i][j], qy[i][j], qyt[i][j]);}\n',
                       '    }\n',

                       # cross section of aerosols
                       '    double *crossp1, *crossp2, *crossp3;\n',
                       '    double crossw1[NLAMBDA], crossw2[NLAMBDA], crossw3[NLAMBDA];\n',
                       '    fp=fopen(AERRADFILE1,"r");\n',
                       '    fp1=fopen(AERRADFILE1,"r");\n',
                       '    s=LineNumber(fp, 1000);\n',
                       '    wavep=dvector(0,s-1);\n',
                       '    crossp1=dvector(0,s-1);\n',
                       '    crossp2=dvector(0,s-1);\n',
                       '    crossp3=dvector(0,s-1);\n',
                       '    k=0;\n',
                       '    while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '        sscanf(dataline, "%lf %lf %lf %lf", wavep+k, crossp1+k, crossp2+k, crossp3+k);\n',
                       '        k=k+1; \n',
                       '    }\n',
                       '    fclose(fp);\n',
                       '    fclose(fp1);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw1, wavep, crossp1, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw2, wavep, crossp2, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw3, wavep, crossp3, s, 0);\n',
                       '    free_dvector(wavep,0,s-1);\n',
                       '    free_dvector(crossp1,0,s-1);\n',
                       '    free_dvector(crossp2,0,s-1);\n',
                       '    free_dvector(crossp3,0,s-1);\n',
                       '    for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '        crossa[1][i] = crossw1[i];\n',
                       '        sinab[1][i]  = crossw2[i]/(crossw1[i]+1.0e-24);\n',
                       '        asym[1][i]   = crossw3[i];\n',
                       '    }\n',
                       '    fp=fopen(AERRADFILE2,"r");\n',
                       '    fp1=fopen(AERRADFILE2,"r");\n',
                       '    s=LineNumber(fp, 1000);\n',
                       '    wavep=dvector(0,s-1);\n',
                       '    crossp1=dvector(0,s-1);\n',
                       '    crossp2=dvector(0,s-1);\n',
                       '    crossp3=dvector(0,s-1);\n',
                       '    k=0;\n',
                       '    while (fgets(dataline, 1000, fp1) != NULL ) {\n',
                       '        sscanf(dataline, "%lf %lf %lf %lf", wavep+k, crossp1+k, crossp2+k, crossp3+k);\n',
                       '        k=k+1; \n',
                       '    }\n',
                       '    fclose(fp);\n',
                       '    fclose(fp1);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw1, wavep, crossp1, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw2, wavep, crossp2, s, 0);\n',
                       '    Interpolation(wavelength, NLAMBDA, crossw3, wavep, crossp3, s, 0);\n',
                       '    free_dvector(wavep,0,s-1);\n',
                       '    free_dvector(crossp1,0,s-1);\n',
                       '    free_dvector(crossp2,0,s-1);\n',
                       '    free_dvector(crossp3,0,s-1);\n',
                       '    for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '        crossa[2][i] = crossw1[i];\n',
                       '        sinab[2][i]  = crossw2[i]/(crossw1[i]+1.0e-24);\n',
                       '        asym[2][i]   = crossw3[i];\n',
                       '    }\n',
                       #    printf("%s\\n", "Cross sections of the aerosol are imported.");\n',
                       #    fprintf(fcheck, "%s\\n", "Cross sections of the aerosol are imported.");\n',
                       #    for (j=0; j<NLAMBDA;j++) {fprintf(fcheck, "%lf %e %e %f %f %f %f\\n", wavelength[j], crossa[1][j], crossa[2][j], sinab[1][j], sinab[2][j], asym[1][j], asym[2][j]);}\n',
                       #    fclose(fcheck);\n',

                       '    FILE *fim;\n',
                       '    double lll[324], ccc[324];\n',
                       '    lll[0] = 400.0;\n',
                       '    for (i=1; i < 324; i++) {\n',
                       '        lll[i]=lll[i-1] * (1.0+1.0 / 200.0);\n',
                       '    }\n',

                       '    char outaer1[1024];\n',
                       '    strcpy(outaer1, OUT_DIR);\n',
                       '    strcat(outaer1, "cross_H2O.dat");\n',
                       '    fim=fopen(outaer1,"r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i < 324; i++) {fscanf(fim, "%le", ccc+i);}\n',
                       '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '            Interpolation( & wavelength[i], 1, & cH2O[j][i], lll, ccc, 324, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',

                       '    char outaer3[1024];\n',
                       '    strcpy(outaer3, OUT_DIR);\n',
                       '    strcat(outaer3, "geo_H2O.dat");\n',
                       '    fim = fopen(outaer3, "r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i < 324; i++) {fscanf(fim, "%lf", ccc+i);}\n',
                       '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '            Interpolation( & wavelength[i], 1, & gH2O[j][i], lll, ccc, 324, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',

                       '    char outaer5[1024];\n',
                       '    strcpy(outaer5, OUT_DIR);\n',
                       '    strcat(outaer5, "albedo_H2O.dat");\n',
                       '    fim = fopen(outaer5, "r");\n',
                       '    for (j=1; j<=zbin; j++) {\n',
                       '        for (i=0; i < 324; i++) {fscanf(fim, "%lf", ccc+i);}\n',
                       '        for (i=' + str(iniz) + '; i<' + str(fine) + '; i++) {\n',
                       '            Interpolation( & wavelength[i], 1, & aH2O[j][i], lll, ccc, 324, 2);\n',
                       '        }\n',
                       '    }\n',
                       '    fclose(fim);\n',

                       # Geometric Albedo 9-point Gauss Quadruture
                       '    double cmiu[9]={-0.9681602395076261,-0.8360311073266358,-0.6133714327005904,-0.3242534234038089,0.0,0.3242534234038089,0.6133714327005904,0.8360311073266358,0.9681602395076261};\n',
                       '    double wmiu[9]={0.0812743883615744,0.1806481606948574,0.2606106964029354,0.3123470770400029,0.3302393550012598,0.3123470770400029,0.2606106964029354,0.1806481606948574,0.0812743883615744};\n',

                       '    double phase;\n',
                       '    phase = ' + str(self.param['phi']) + ';\n',  # Phase Angle, 0 zero geometric albedo
                       '    double lonfactor1, lonfactor2;\n',
                       '    double latfactor1, latfactor2;\n',
                       '    lonfactor1 = (PI-phase)*0.5;\n',
                       '    lonfactor2 = phase*0.5;\n',
                       '    latfactor1 = PI*0.5;\n',
                       '    latfactor2 = 0;\n',

                       '    double lat[9], lon[9];\n',
                       '    for (i=0; i<9; i++) {\n',
                       '        lat[i] = latfactor1*cmiu[i]+latfactor2;\n',
                       '        lon[i] = lonfactor1*cmiu[i]+lonfactor2;\n',
                       '    }\n',
                       '    double T0[zbin + 1];\n',
                       '    for (j=0; j <= zbin; j++) {\n',
                       '        T0[j]=0.0;\n',
                       '    }\n',

                       '    char uvrfile[1024];\n',

                       # Variation
                       '    double methaneexp[7]={0,0,0,0,0,0,0};\n',
                       '    int methaneid;\n',

                       '    double gmiu0, gmiu;\n',
                       '    double rout[NLAMBDA], gal[NLAMBDA];\n',
                       '    for (k=' + str(iniz) + '; k < ' + str(fine) + '; k++) {\n',
                       '        gal[k]=0;\n',
                       '    }\n',

                       '    for (j=1; j <= zbin; j++) {\n',
                       '        for (i=1; i <= NSP; i++) {\n',
                       '            xx1[j][i] = xx[j][i];\n',
                       '        }\n',
                       '    }\n',

                       '    strcpy(uvrfile, OUT_DIR);\n',
                       '    strcat(uvrfile, "Reflection_Phase.dat");\n',
                       '    for (i=0; i < 9; i++) {\n',
                       '        for (j=0; j < 9; j++) {\n',
                       '            gmiu0 = cos(lat[i]) * cos(lon[j]-phase);\n',
                       '            gmiu  = cos(lat[i]) * cos(lon[j]);\n',
                       '            if (fabs(gmiu0-gmiu) < 0.0000001) {\n',
                       '                gmiu=gmiu0+0.0000001;\n',
                       '            }\n',
                       # printf("%f %f %f %f\n", lat[i], lon[j], gmiu0, gmiu);
                       '            Reflection(xx1, T, stdcross, qysum, cross, crosst, uvrfile, gmiu0, gmiu, phase, rout, ' + str(iniz) + ', ' + str(fine) + ');\n',
                       '            for (k=' + str(iniz) + '; k < ' + str(fine) + '; k++) {\n',
                       '                gal[k] += wmiu[i] * wmiu[j] * rout[k] * gmiu0 * gmiu * cos(lat[i]) * latfactor1 * lonfactor1 / PI;\n',
                       '            }\n',
                       '        }\n',
                       '    }\n',

                       # print out spectra
                       '    char outag[1024];\n',
                       '    strcpy(outag, OUT_DIR);\n',
                       '    strcat(outag, "PhaseA.dat");\n',
                       '    fp = fopen(outag, "w");\n',
                       '    for (i=' + str(iniz) + '; i < ' + str(fine) + '; i++) {\n',
                       '        fprintf(fp, "%f\t", wavelength[i]);\n',
                       '        fprintf(fp, "%e\t", gal[i]);\n',
                       '        fprintf(fp, "\\n");\n',
                       '    }\n',
                       '    fclose(fp);\n',

                       # Clean up
                       '    free_dmatrix(cross, 1, nump, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(qy, 1, nump, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacH2O, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacNH3, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacCH4, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacH2S, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacSO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacCO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacCO, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacO3, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacN2O, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(opacN2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacOH, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacH2CO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacH2O2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacC2H2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacC2H4, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacC2H6, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHCN, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCH2O2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHNO3, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacNO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacNO2, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacOCS, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHF, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHCl, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHBr, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHI, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacClO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHClO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacHBrO, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacPH3, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCH3Cl, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCH3Br, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacDMS, 1, zbin, 0, NLAMBDA - 1);\n',
                       #    free_dmatrix(opacCS2, 1, zbin, 0, NLAMBDA - 1);\n',
                       '    free_dmatrix(xx, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx1, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx2, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx3, 1, zbin, 1, NSP);\n',
                       '    free_dmatrix(xx4, 1, zbin, 1, NSP);\n',

                       '}\n']

        with open(self.c_code_directory + 'core_' + str(self.process) + '.c', 'w') as file:
            for riga in c_core_file:
                file.write(riga)

    def __run_c_code(self):
        self.__par_c_file()
        self.__core_c_file()
        os.chdir(self.c_code_directory)
        if platform.system() == 'Darwin':
            os.system('clang -Wno-nullability-completeness -o ' + str(self.process) + ' core_' + str(self.process) + '.c -lm')
        else:
            os.system('gcc -o ' + str(self.process) + ' core_' + str(self.process) + '.c -lm')
        while not os.path.exists(self.c_code_directory + str(self.process)):
            pass
        time.sleep(2)
        os.system('chmod +rwx ' + str(self.process))
        os.system('./' + str(self.process))
        os.chdir(self.working_dir)
        time1 = time.time()
        broken = False
        while not os.path.exists(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/PhaseA.dat'):
            time2 = time.time()
            if time2 - time1 > 600:
                broken = True
                break
            else:
                pass
        os.system('rm -rf ' + self.c_code_directory + str(self.process))
        os.system('rm -rf ' + self.c_code_directory + 'core_' + str(self.process) + '.c')
        os.system('rm -rf ' + self.c_code_directory + 'par_' + str(self.process) + '.h')
        if not broken:
            albedo = np.loadtxt(self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/PhaseA.dat')
        else:
            albedo = np.zeros((1000, 2))
            albedo[:, 1] = np.nan
            albedo[:, 0] = np.linspace(self.param['min_wl'], self.param['max_wl'], num=1000)
        if self.retrieval:
            os.system('rm -rf ' + self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
        else:
            if self.canc_metadata:
                os.system('rm -rf ' + self.c_code_directory + 'Result/Retrieval_' + str(self.process) + '/')
            else:
                pass

        return albedo[:, 0], albedo[:, 1]

    def run_forward(self):
        self.__run_structure()
        alb_wl, alb = self.__run_c_code()

        return alb_wl, alb


def forward(parameters_dictionary, evaluation=None, phi=None, n_obs=None, retrieval_mode=True, core_number=None, albedo_calc=False, fp_over_fs=False, canc_metadata=False):
    param = copy.deepcopy(parameters_dictionary)

    if evaluation is not None:
        if not param['rocky']:
            param['vmr_H2O'] = evaluation['vmr_H2O']
            param['vmr_NH3'] = evaluation['vmr_NH3']
            param['vmr_CH4'] = evaluation['vmr_CH4']
            if param['fit_wtr_cld']:
                param['Pw_top'] = evaluation['pH2O']
                param['cldw_depth'] = evaluation['dH2O']
                param['CR_H2O'] = evaluation['crH2O']
            if param['fit_amm_cld']:
                param['Pa_top'] = evaluation['pNH3']
                param['clda_depth'] = evaluation['dNH3']
                param['CR_NH3'] = evaluation['crNH3']
        else:
            if param['fit_p0'] or param['gas_par_space'] == 'partial_pressure':
                param['P0'] = evaluation['P0']
            if param['fit_wtr_cld']:
                param['Pw_top'] = evaluation['pH2O']
                param['cldw_depth'] = evaluation['dH2O']
                param['CR_H2O'] = evaluation['crH2O']

            for mol in param['fit_molecules']:
                param['vmr_' + mol] = evaluation[mol]
            if param['gas_fill'] is not None:
                param['vmr_' + param['gas_fill']] = evaluation[param['gas_fill']]

            if param['fit_ag']:
                if param['surface_albedo_parameters'] == int(1):
                    param['Ag'] = evaluation['ag']
                elif param['surface_albedo_parameters'] == int(3):
                    for surf_alb in [1, 2]:
                        param['Ag' + str(surf_alb)] = evaluation['ag' + str(surf_alb)]
                    param['Ag_x1'] = evaluation['ag_x1']
                elif param['surface_albedo_parameters'] == int(5):
                    for surf_alb in [1, 2, 3]:
                        param['Ag' + str(surf_alb)] = evaluation['ag' + str(surf_alb)]
                    param['Ag_x1'] = evaluation['ag_x1']
                    param['Ag_x2'] = evaluation['ag_x2']


            if param['fit_T']:
                param['Tp'] = evaluation['Tp']
        if param['fit_g'] and param['fit_Mp'] and not param['fit_Rp']:
            param['gp'] = (10. ** (evaluation['gp'] - 2.0))                                                                     # g is in m/s2 but it was defined in cgs
            param['Mp'] = evaluation['Mp']                                                                                      # Mp is in M_jup
            param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * param['Mp']) / param['gp'])) / const.R_jup.value        # Rp is in R_jup
        elif param['fit_g'] and param['fit_Rp'] and not param['fit_Mp']:
            param['gp'] = (10. ** (evaluation['gp'] - 2.0))                                                                     # g is in m/s2 but it was defined in cgs
            param['Rp'] = evaluation['Rp']                                                                                      # Rp is in R_jup
            param['Mp'] = ((param['gp'] * ((param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value       # Mp is in M_jup
        elif param['fit_Mp'] and param['fit_Rp'] and not param['fit_g']:
            param['Mp'] = evaluation['Mp']                                                                                      # Mp is in M_jup
            param['Rp'] = evaluation['Rp']                                                                                      # Rp is in R_jup
            param['gp'] = (const.G.value * const.M_jup.value * param['Mp']) / ((const.R_jup.value * param['Rp']) ** 2.)         # g is in m/s2
        elif param['fit_g'] and not param['fit_Mp'] and not param['fit_Rp'] and param['Mp'] is not None:
            param['gp'] = (10. ** (evaluation['gp'] - 2.0))                                                                     # g is in m/s2 but it was defined in cgs
            param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * param['Mp']) / param['gp'])) / const.R_jup.value        # Rp is in R_jup
        elif param['fit_Rp'] and not param['fit_Mp'] and not param['fit_g'] and param['Mp'] is not None:
            param['Rp'] = evaluation['Rp']                                                                                      # Rp is in R_jup
            if param['Mp_err'] is not None and param['Mp_prior_type'] == 'random_error':
                param['Mp'] = np.random.normal(param['Mp_orig'], param['Mp_err'])
            else:
                param['Mp'] = param['Mp_orig'] + 0.0
            param['gp'] = (const.G.value * const.M_jup.value * param['Mp']) / ((const.R_jup.value * param['Rp']) ** 2.)         # g is in m/s2
        elif not param['fit_g'] and not param['fit_Mp'] and not param['fit_Rp']:
            if not param['Mp_provided']:
                param['Mp'] = ((param['gp'] * ((param['Rp'] * const.R_jup.value) ** 2.)) / const.G.value) / const.M_jup.value   # Mp is in M_jup
            if not param['Rp_provided']:
                param['Rp'] = (np.sqrt((const.G.value * const.M_jup.value * param['Mp']) / param['gp'])) / const.R_jup.value    # Rp is in R_jup
            if not param['gp_provided']:
                param['gp'] = (const.G.value * const.M_jup.value * param['Mp']) / ((const.R_jup.value * param['Rp']) ** 2.)     # g is in m/s2

        if param['fit_p_size']:
            param['p_size'] = evaluation['p_size']

        if param['fit_phi']:
            param['phi'] = evaluation['phi']

    if phi is not None:
        param['phi'] = phi

    param['core_number'] = core_number

    if not param['rocky']:
        param = cloud_pos(param)
        param = calc_mean_mol_mass(param)
        mod = FORWARD_GAS_MODEL(param, retrieval=retrieval_mode, canc_metadata=canc_metadata)
    else:
        if param['gas_par_space'] == 'partial_pressure' and np.log10(param['P0']) < 0.0:
            param['P0'] = 1.1
        param['P'] = 10. ** np.arange(0.0, np.log10(param['P0']) + 0.01, step=0.01)
        param['vmr_H2O'] = cloud_rocky_pos(param)
        param = adjust_VMR(param, all_gases=param['adjust_VMR_gases'])
        if param['O3_earth']:
            param['vmr_O3'] = ozone_earth_mask(param)
        param = calc_mean_mol_mass(param)
        mod = FORWARD_ROCKY_MODEL(param, retrieval=retrieval_mode, canc_metadata=canc_metadata)

    alb_wl, alb = mod.run_forward()
    alb_wl *= 10. ** (-3.)

    if not retrieval_mode:
        if n_obs == 0 or n_obs is None:
            if albedo_calc and not fp_over_fs:
                print('Calculating the planetary albedo as function of wavelength')
            elif fp_over_fs and not albedo_calc:
                print('Calculating the contrast ratio as function of wavelength')
            else:
                print('Calculating the planetary flux as function of wavelength')

    wl, model = model_finalizzation(param, alb_wl, alb, planet_albedo=albedo_calc, fp_over_fs=fp_over_fs, n_obs=n_obs)

    if retrieval_mode:
        return model
    else:
        return wl, model

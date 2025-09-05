# color palette: #01b8aa,#003f5c,#20639b,#ffa600,#ed553b

import matplotlib.pyplot as plt
import numpy as np

def NGA_test_plot_rd():
    # JPEG2000
    jp2k_bpp                = [2.6647 , 1.999  , 1.5992 , 0.9996 ]#, 0.5323]
    jp2k_amp_psnr           = [28.925 , 24.4725, 22.2939, 18.6632]#, 15.3867]
    jp2k_amp_sqnr           = [18.4845, 14.1511, 11.9136, 8.3596 ]#, 5.1653]
    jp2k_phase_mape         = [0.3918 , 0.6234 , 0.8121 , 1.1672 ]#, 1.5330]

    # HEVC test
    hevc_qps                = [6      , 8      , 12     , 16     , 20]
    HEVC_bpp                = [2.8405 , 2.5788 , 1.9839 , 1.4525 , 0.9826 ]
    HEVC_amp_psnr           = [33.1111, 31.5369, 27.8776, 24.1401, 20.6104]
    HEVC_amp_sqnr           = [22.6497, 21.0718, 17.4426, 13.7223, 10.2396]
    HEVC_phase_mape         = [0.2195 , 0.2593 , 0.3812 , 0.5717 , 0.8370]

    # LearnedIQ test
    learnedIQ_qp             = [6     , 8      , 12     , 16     , 20]
    learnedIQ_bpp           = [2.8405 , 2.5788 , 1.9839 , 1.4525 , 0.9826 ]
    learnedIQ_amp_psnr      = [34.3395, 32.9682, 29.4298, 25.3870, 21.5924]
    learnedIQ_amp_sqnr      = [23.9356, 22.5431, 18.9904, 14.9526, 11.1809]
    learnedIQ_phase_mape    = [0.1894 , 0.2232 , 0.3366 , 0.5202 , 0.7773]

    # VVC test
    vvc_qp                  = [1      , 3      , 7      , 11     , 16     ]#, 19]
    VVC_uniform_bpp         = [2.8008 , 2.5111 , 1.98   , 1.4989 , 0.9148 ]#, 0.5456 ]
    VVC_uniform_amp_psnr    = [34.4543, 32.8780, 29.3566, 25.9315, 21.1603]#, 17.3979]
    VVC_uniform_amp_sqnr    = [23.9722, 22.4221, 18.9250, 15.5249, 10.7797]#, 7.0912]
    VVC_uniform_phase_mape  = [0.1867 , 0.2199 , 0.3190 , 0.4632 , 0.7914 ]#, 1.1828]

    lowlatency_P_bpp        = [5.9549, 5.6095 , 5.0285 , 3.9599 , 2.9913, 1.8249]   # [1.2076, 1.9444, 4.0894,  5.1761,  5.7806, 6.0116]#[5.6538]#[5.4713]
    lowlatency_P_psnr       = [35.481, 34.4508, 32.8747, 29.3292, 25.935, 21.185]   # [18.0910, 21.6586, 29.7701, 33.2245, 34.9387, 35.6132]#[34.1076]
    lowlatency_P_phase_mape = [0.1681, 0.1891 , 0.2203 , 0.3214 , 0.4645, 0.7645]
    
    # VVC on the IQ and computed amplitude from AmpRes
    VVC_ampres_bpp          = [2.8008 , 2.5111 , 1.9800 , 1.4989 , 0.9148 ]#, 0.5456]
    VVC_ampres_psnr         = [36.0029, 34.4068, 30.6787, 27.0996, 22.5055]#, 19.7884]
    VVC_ampres_sqnr         = [25.5335, 23.9561, 20.2304, 16.6512, 12.1022]#, 9.427]
    VVC_angres_phase_mape   = [0.1559 , 0.1854 , 0.2828 , 0.4216 , 0.7495 ]#, 1.1503]

    sarelic_lambda          = [35     , 25     , 15     , 5      , 0.9]
    sarelic_mse_bpp         = [2.8954 , 2.7357 , 2.4915 , 2.0025 , 1.2981]
    sarelic_mse_amp_psnr    = [39.4463, 38.8709, 37.6056, 34.1411, 27.6849]
    sarelic_mse_amp_sqnr    = [30.8424, 29.8034, 28.0426, 23.9668, 17.2689]
    sarelic_mse_phase_mape  = [0.08259, 0.0932 , 0.1141 , 0.1843 , 0.3929]

    sarelic_l1_bpp          = [2.6126 , 2.3683 , 2.0308 , 1.7651 , 1.5290]
    sarelic_l1_amp_psnr     = [38.0279, 36.7034, 34.4309, 32.2751, 30.0564]
    sarelic_l1_phase_mape   = [0.0947 , 0.1171 , 0.1602 , 0.2141 , 0.27943]

    sarelic_lambda           = [35     , 25     , 15     , 10     , 4      , 1      ]
    sarelic_dct_v2_bpp       = [2.9640 , 2.7782 , 2.5324 , 2.3432 , 1.9434 , 1.3604 ] # mse
    sarelic_dct_v2_psnr      = [39.0820, 38.6299, 37.3533, 36.2189, 33.2039, 27.8906] # mse
    sarelic_dct_v2_sqnr      = [30.6782, 29.6883, 27.9480, 26.5056, 23.0530, 17.5115]
    sarelic_dct_v2_phase_mape= [0.0833 , 0.0936 , 0.1150 , 0.1364 , 0.2037 , 0.3796 ] # mse

    sarelic_entropy_group_bpp = [2.0951]
    sarelic_entropy_group_psnr = [33.39]
    sarelic_entropy_group_sqnr = [23.1806]
    sarelic_entropy_group_mape = [0.2013]

    sarelic_ampres_bpp  = [2.3432  ,1.9434]
    sarelic_ampres_psnr = [36.2978 ,33.2781]
    sarelic_ampres_sqnr = [26.5560 ,23.1250]

    # plt.figure();plt.plot(np.array(sarelic_dct_v2_bpp[3:5]), sarelic_dct_v2_sqnr[3:5], color = '#ed553b', marker="*")
    # plt.plot(np.array(sarelic_ampres_bpp), sarelic_ampres_sqnr, color = 'r', linestyle="dotted",marker="D")
    # plt.ylabel("sqnr")
    # plt.xlabel("bpp")
    # plt.legend(["E2E", "E2E+AmpRes"])
    # plt.grid()
    # plt.figure();plt.plot(np.array(sarelic_dct_v2_bpp[3:5]), sarelic_dct_v2_psnr[3:5], color = '#ed553b', marker="*")
    # plt.plot(np.array(sarelic_ampres_bpp), sarelic_ampres_psnr, color = 'r', linestyle="dotted",marker="D")
    # plt.ylabel("psnr")
    # plt.xlabel("bpp")
    # plt.legend(["E2E", "E2E+AmpRes"])
    # plt.grid()

    sarelic_mst_dct_bpp       = [1.923]
    sarelic_mst_dct_psnr      = [32.9909]
    sarelic_mst_dct_sqnr      = [22.9519]
    sarelic_mst_dct_phase_mape= [0.2051]

    sarelic_mst_dctv2_bpp       = [1.9251 ,2.5145 ,2.931]
    sarelic_mst_dctv2_psnr      = [33.0124,36.8057,38.1631]
    sarelic_mst_dctv2_sqnr      = [22.9671,27.7438,30.3835]
    sarelic_mst_dctv2_phase_mape= [0.2031 ,0.1168 ,0.0852]

    sarelic_nmse_bpp        = [2.7107, 2.9011]
    sarelic_nmse_amp_psnr   = [37.1908, 37.8478]
    sarelic_nmse_phase_mape = [0.1088, 0.0951]

    sarelic_iq_scaled_bpp        = [3.033, 2.692]
    sarelic_iq_scaled_amp_psnr   = [39.389, 38.65]
    sarelic_iq_scaled_phase_mape = [0.0773, 0.0975]

    # RD-curve AMP PSNR 
    plt.figure()
    plt.plot(np.array(jp2k_bpp), jp2k_amp_psnr, color = '#01b8aa', marker='^')
    plt.plot(np.array(HEVC_bpp), HEVC_amp_psnr, color = '#003f5c',marker="d")
    plt.plot(np.array(learnedIQ_bpp), learnedIQ_amp_psnr, color = '#20639b',linestyle="dashed",marker="P")
    plt.plot(np.array(VVC_uniform_bpp), VVC_uniform_amp_psnr, color = '#ffa600',marker="s")
    plt.plot(np.array(VVC_ampres_bpp), VVC_ampres_psnr, color = '#ffa600',marker=".", linestyle="dotted")
    plt.plot(np.array(sarelic_dct_v2_bpp), sarelic_dct_v2_psnr, color = '#ed553b', marker="*")
    plt.plot(np.array(sarelic_entropy_group_bpp), sarelic_entropy_group_psnr, color = 'r', marker="o")
    #plt.plot(np.array(sarelic_l1_bpp), sarelic_l1_amp_psnr, color = 'r', linestyle="dotted",marker="D")
    #plt.plot(np.array(sarelic_mse_bpp), sarelic_mse_amp_psnr, color = 'b', linestyle="dotted",marker="P")
    # plt.plot(np.array(sarelic_mst_dct_bpp), sarelic_mst_dct_psnr, color = 'r', linestyle="dotted",marker="D")
    # plt.plot(np.array(sarelic_mst_dctv2_bpp), sarelic_mst_dctv2_psnr, color = 'b', linestyle="dotted",marker=".")
    # plt.plot([1.33, 1.87], [27.9914, 33.1192], marker="o", color = 'r')

    plt.title("BPP vs PSNR RD-curve for NGA test sequence amplitude image")
    plt.xlabel("Bits per pixel (bpp) per band")
    plt.ylabel("PSNR (dB)")
    plt.legend(["JPEG2000", "HEVC(Intra)", "Maharjan[1]", "VVC(Intra)", "AmpRes", "E2E-DCT(Ours)", "E2E-Entropy(Ours)"])
    #plt.legend(["JPEG2000", "LearnedIQ-HEVC", "AmpRes-VVC", "Ours"])
    plt.grid()
    #plt.savefig("./gimp/rd/nga_rd_psnr.png")
    
    #RD-curve AMP SQNR
    plt.figure()
    plt.plot(np.array(jp2k_bpp), jp2k_amp_sqnr, color = '#01b8aa', marker='^')
    plt.plot(np.array(HEVC_bpp), HEVC_amp_sqnr, color = '#003f5c',marker="d")
    plt.plot(np.array(learnedIQ_bpp), learnedIQ_amp_sqnr, color = '#20639b',linestyle="dashed", marker="P")
    plt.plot(np.array(VVC_uniform_bpp), VVC_uniform_amp_sqnr, color = '#ffa600',marker="s")
    plt.plot(np.array(VVC_ampres_bpp), VVC_ampres_sqnr, color = '#ffa600',marker=".", linestyle="dotted")
    plt.plot(np.array(sarelic_dct_v2_bpp), sarelic_dct_v2_sqnr, color = '#ed553b', marker="*")
    plt.plot(np.array(sarelic_entropy_group_bpp), sarelic_entropy_group_sqnr, color = 'r', marker="o")
    # plt.plot(np.array(sarelic_mst_dct_bpp), sarelic_mst_dct_sqnr, color = 'r', linestyle="dotted",marker="D")
    # plt.plot(np.array(sarelic_mst_dctv2_bpp), sarelic_mst_dctv2_sqnr, color = 'b', linestyle="dotted",marker=".")

    plt.title("BPP vs SQNR RD-curve for NGA test sequence amplitude image")
    plt.xlabel("Bits per pixel (bpp) per band")
    plt.ylabel("SQNR (dB)")
    plt.legend(["JPEG2000", "HEVC(Intra)", "Maharjan[1]", "VVC(Intra)", "AmpRes", "E2E-DCT(Ours)", "E2E-Entropy(Ours)"])
    #plt.legend(["JPEG2000", "LearnedIQ-HEVC", "AmpRes-VVC", "Ours"])
    plt.grid()
    #plt.savefig("./gimp/rd/nga_rd_sqnr.png")


    # RD-curve Phase MAPE 
    plt.figure()
    plt.plot(np.array(jp2k_bpp), jp2k_phase_mape, color = '#01b8aa', marker='^')
    plt.plot(np.array(HEVC_bpp), HEVC_phase_mape, color = '#003f5c',marker="d")
    plt.plot(np.array(learnedIQ_bpp), learnedIQ_phase_mape, color = '#20639b', linestyle="dashed", marker="P")
    plt.plot(np.array(VVC_uniform_bpp), VVC_uniform_phase_mape, color = '#ffa600',marker="s")
    plt.plot(np.array(VVC_ampres_bpp), VVC_angres_phase_mape, color = '#ffa600',marker=".", linestyle="dotted")
    plt.plot(np.array(sarelic_dct_v2_bpp), sarelic_dct_v2_phase_mape, color = '#ed553b', marker="*")
    plt.plot(np.array(sarelic_entropy_group_bpp), sarelic_entropy_group_mape, color = 'r', marker="o")
    # plt.plot(np.array(sarelic_mst_dct_bpp), sarelic_mst_dct_phase_mape, color = 'r', linestyle="dotted",marker="D")
    # plt.plot(np.array(sarelic_mst_dctv2_bpp), sarelic_mst_dctv2_phase_mape, color = 'b', linestyle="dotted",marker=".")
     
    plt.title("BPP vs MAPE RD-curve for NGA test sequence phase image\n(Smaller area under curve is better for MAPE.)")
    plt.xlabel("Bits per pixel (bpp) per band")
    plt.ylabel("MAPE (rad)")
    plt.legend(["JPEG2000", "HEVC(Intra)", "Maharjan[1]","VVC(Intra)", "AngRes-DCT", "E2E-DCT(Ours)", "E2E-Entropy(Ours)"])
    # plt.legend(["JPEG2000", "LearnedIQ-HEVC","AngRes-DCT-VVC", "Ours"])
    plt.grid()
    #plt.savefig("./gimp/rd/nga_rd_mape.png")

if __name__ == "__main__":
    NGA_test_plot_rd()
    plt.show()
    print("Done...")
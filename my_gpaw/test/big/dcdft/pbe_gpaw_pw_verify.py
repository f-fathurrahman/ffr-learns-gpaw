import csv

from my_gpaw.test.big.dcdft.pbe_gpaw_pw import tag

# calculated with pw=1200, k-point-density=8, width=0.06, fit=(5, 0.02)
data = """
H,17.459,10.34,2.705
He,17.264,1.047,6.06
Li,20.259,14.058,3.365
Be,8.008,123.494,3.348
B,7.241,237.967,3.464
C,11.645,209.686,3.566
N,28.797,54.159,3.694
O,18.525,51.715,3.895
F,19.049,37.197,3.735
Ne,23.938,1.43,6.456
Na,37.321,7.609,-1.136
Mg,22.892,36.231,4.033
Al,16.517,77.66,4.609
Si,20.52,88.516,4.326
P,21.532,68.002,4.348
S,17.187,84.008,4.207
Cl,38.949,19.016,4.378
Ar,52.201,0.788,7.033
K,73.563,3.608,3.772
Ca,42.395,17.422,3.273
Sc,24.573,54.613,3.393
Ti,17.437,112.194,3.594
V,13.543,182.429,3.802
Cr,11.852,161.406,7.241
Mn,11.471,121.43,0.493
Fe,11.472,193.38,5.62
Co,10.91,215.828,5.115
Ni,10.982,203.94,4.947
Cu,12.081,137.188,5.078
Zn,15.122,76.379,5.407
Ga,20.538,49.728,5.378
Ge,23.967,60.21,4.906
As,22.637,68.795,4.294
Se,29.761,47.304,4.481
Br,39.768,22.302,4.866
Kr,66.26,0.636,7.271
Rb,91.121,2.786,3.765
Sr,54.662,12.391,4.328
Y,32.858,41.22,3.014
Zr,23.323,93.99,3.322
Nb,18.034,170.426,3.92
Mo,15.765,259.394,4.367
Ru,13.735,309.531,4.858
Rh,14.137,253.502,5.26
Pd,15.304,169.253,5.614
Ag,17.807,92.333,5.751
Cd,22.575,46.346,6.714
In,27.244,36.05,5.299
Sn,36.581,36.313,4.912
Sb,31.593,50.484,4.485
Te,34.708,45.262,4.716
I,50.594,18.551,5.083
Xe,87.204,0.529,7.235
Cs,117.31,1.961,3.492
Ba,63.575,8.813,2.853
Hf,22.586,108.63,3.452
Ta,18.302,193.805,3.813
W,16.115,304.826,4.245
Re,14.926,365.244,4.435
Os,14.194,400.169,4.813
Ir,14.452,348.912,5.111
Pt,15.613,246.217,5.496
Au,18.208,138.897,5.823
Hg,28.515,10.958,11.346
Tl,31.107,27.488,5.364
Pb,31.872,37.943,4.848
Bi,36.831,42.751,4.64
Rn,93.543,0.523,7.228
"""

names = [r.split(',')[0] for r in data.split()][1:]
ref = {}
for name in names:
    for l in data.split():
        if l.split(',')[0] == name:
            ref[name] = [float(v) for v in l.split(',')[1:]]

csvreader = csv.reader(open(tag + '_raw.csv', 'rb'))
calc = {}
for r in csvreader:
    if "#" not in r[0]:
        calc[r[0]] = [float(v) for v in r[1:]]

for name in names:
    if name in calc.keys():
        vref = ref[name][0]
        vcalc = calc[name][0]
        errmsg = name + ': ' + str(vref) + ' vs ' + str(vcalc)
        assert abs(vcalc - vref) / vref < 0.002, errmsg
        b0ref = ref[name][1]
        b0calc = calc[name][1]
        errmsg = name + ': ' + str(b0ref) + ' vs ' + str(b0calc)
        assert abs(b0calc - b0ref) / b0ref < 0.01, errmsg
        b1ref = ref[name][2]
        b1calc = calc[name][2]
        errmsg = name + ': ' + str(b1ref) + ' vs ' + str(b1calc)
        assert abs(b1calc - b1ref) / b1ref < 0.05, errmsg

from my_df import DielectricFunction

# Part 2 : Spectrum calculation
# DF: dielectric function object
# Ground state gpw file (with wavefunction) as input
df = DielectricFunction(
    calc="Si_gs.gpw",
    frequencies={
        "type": "nonlinear",
        "domega0": 0.05
    },  # using nonlinear frequency grid
    rate="eta")

# By default, a file called "df.csv" is generated
#df.get_dielectric_function(direction="x", filename="TEMP_eps_x.csv")

#res_inv_diel_func = df.get_inverse_dielectric_function(
#    direction="x", truncation=df.truncation) #.macroscopic_dielectric_function()

#chi0 = df.get_chi0_dyson_eqs() #.inverse_dielectric_function(direction=direction)
# Chi0DysonEquations

from debug_diel_funcs import debug_get_chi0_dyson_eqs
debug_get_chi0_dyson_eqs(df)


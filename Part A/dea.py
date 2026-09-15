import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ============================================================
# 1. ΔΕΔΟΜΕΝΑ ΒΑΣΗΣ
# ============================================================

# Στήλες:
# M1 = Μέση ημερήσια κάλυψη
# M2 = Μέση διάρκεια ακρόασης
# M3 = Μερίδιο ακροαματικότητας
# M4 = Έσοδα από διαφημίσεις
# M5 = Αριθμός εργαζομένων
# M6 = Μηνιαίο λειτουργικό κόστος
# M7 = Αναμεταδόσεις

data = [
    ["ALPHA 98.9",          1.7, 79.0, 0.8, 1109000, 100, 195870, 24],
    ["ATHENS DJ",           8.3, 76.1, 3.6, 1250000, 20, 47300, 10],
    ["BEST 92.6",           3.8, 120.3, 2.5, 1503250, 27, 91300, 11],
    ["ΔΙΕΣΗ 101.3",         7.2, 84.4, 3.4, 1125900, 35, 53980, 1],
    ["ΔΡΟΜΟΣ FM",           6.9, 65.3, 2.5, 990000, 28, 32000, 1],
    ["EASY 97.2",           6.8, 107.0, 4.1, 1200300, 30, 45000, 2],
    ["ΕΛΛΗΝΙΚΟΣ 93.2",      5.5, 95.4, 2.9, 1002300, 27, 32560, 3],
    ["EN LEFKO 87.7",       5.1, 102.6, 2.9, 906000, 21, 29500, 3],
    ["GALAXY 92.0",         3.1, 111.9, 1.9, 1645000, 28, 96000, 1],
    ["HIT 88.9",             5.5, 70.5, 2.2, 1870000, 32, 58000, 5],
    ["KISS FM 92.9",         5.4, 77.8, 2.4, 2124000, 25, 65100, 13],
    ["ΛΑΜΨΗ 92.3",           5.7, 61.2, 2.0, 3301000, 38, 79600, 14],
    ["LOVE RADIO 97.50",     4.8, 62.3, 2.1, 1780900, 30, 42900, 9],
    ["MAD RADIO",            4.2, 61.2, 1.4, 350000, 20, 30260, 3],
    ["ΜΕΛΩΔΙΑ FM 99.2",      8.7, 92.2, 4.5, 2002130, 25, 53600, 10],
    ["MENTA 88.0 FM",        5.4, 90.7, 2.8, 1560000, 22, 36700, 2],
    ["ΜΟΥΣΙΚΟΣ",             4.3, 99.3, 2.4, 1450000, 37, 45600, 3],
    ["MUSIC 89.2",            3.7, 63.1, 1.3, 1350700, 36, 63000, 1],
    ["ΠΑΡΑΠΟΛΙΤΙΚΑ 90.1",    3.6, 131.3, 2.7, 1000200, 45, 67800, 7],
    ["PEPPER 96.6",           5.7, 115.1, 3.7, 1250000, 27, 43700, 4],
    ["REAL FM 97.8",          7.9, 120.4, 5.3, 2180000, 87, 89700, 16],
    ["RED 96.3",              7.8, 70.4, 3.1, 1605100, 21, 45110, 2],
    ["ATHENS ROCK FM",        7.2, 66.9, 2.7, 1007200, 24, 30980, 10],
    ["ΡΥΘΜΟΣ 9 49",           8.2, 69.4, 3.2, 2130600, 28, 49890, 18],
    ["SFERA 102.2",            6.8, 78.8, 3.0, 3105000, 45, 101230, 18],
    ["SKAI 100.3",             9.3, 96.0, 5.0, 4530000, 110, 220900, 31],
    ["ΣΠΟΡ FM 94.6",            6.1, 81.6, 2.8, 2300200, 97, 103890, 15],
    ["SPORT 24 RADIO 103.3",    4.2, 121.4, 2.9, 1890000, 86, 114900, 16],
    ["ΣΤΟ ΚΟΚΚΙΝΟ",             1.9, 130.8, 1.4, 980000, 35, 65300, 5]
]

columns = [
    "Station",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7"
]

df = pd.DataFrame(data, columns=columns)

print("\nΑΡΧΙΚΑ ΔΕΔΟΜΕΝΑ")
print(df)


# 2. MONTE CARLO SIMULATION

# Για αναπαραγωγιμότητα
np.random.seed(42)

N = 1000

# Θα αποθηκεύσουμε όλες τις προσομοιώσεις
simulation_results = []

# Θα αποθηκεύσουμε τους μέσους όρους των 1000 προσομοιώσεων
mean_results = []

for index, row in df.iterrows():

    # M1: Uniform(-0.08, 0.08)

    epsilon1 = np.random.uniform(-0.08, 0.08, N)
    M1_sim = row["M1"] * (1 + epsilon1)

    # M2: Normal(0, 0.05)

    epsilon2 = np.random.normal(0, 0.05, N)
    M2_sim = row["M2"] * (1 + epsilon2)

    # M3: Uniform(-0.10, 0.10)

    epsilon3 = np.random.uniform(-0.10, 0.10, N)
    M3_sim = row["M3"] * (1 + epsilon3)

    # M4: Normal(0, 0.08)

    epsilon4 = np.random.normal(0, 0.08, N)
    M4_sim = row["M4"] * (1 + epsilon4)

    # M5: Poisson(2)-2

    z5 = np.random.poisson(2, N) - 2
    M5_sim = np.round(row["M5"] + z5)

    # M6: Uniform(-0.12, 0.12)

    epsilon6 = np.random.uniform(-0.12, 0.12, N)
    M6_sim = row["M6"] * (1 + epsilon6)

    # M7: Poisson(1)-1

    z7 = np.random.poisson(1, N) - 1
    M7_sim = np.round(row["M7"] + z7)

    # --------------------------------------------------------
    # Αποθήκευση όλων των παρατηρήσεων
    # --------------------------------------------------------

    for i in range(N):

        simulation_results.append([
            row["Station"],
            M1_sim[i],
            M2_sim[i],
            M3_sim[i],
            M4_sim[i],
            M5_sim[i],
            M6_sim[i],
            M7_sim[i]
        ])

    # Μέσος όρος των 1000 παρατηρήσεων


    mean_results.append([
        row["Station"],
        np.mean(M1_sim),
        np.mean(M2_sim),
        np.mean(M3_sim),
        np.mean(M4_sim),
        np.mean(M5_sim),
        np.mean(M6_sim),
        np.mean(M7_sim)
    ])


# DataFrame με όλες τις 29.000 παρατηρήσεις
simulation_df = pd.DataFrame(
    simulation_results,
    columns=columns
)

# Νέο dataset που θα χρησιμοποιηθεί στην ΠΑΔ
mean_df = pd.DataFrame(
    mean_results,
    columns=columns
)

print("\n\nΜΕΣΕΣ ΤΙΜΕΣ MONTE CARLO")
print(mean_df)



# 3. ΠΕΡΙΓΡΑΦΙΚΗ ΣΤΑΤΙΣΤΙΚΗ


statistics = []

for variable in columns[1:]:

    values = simulation_df[variable]

    mean_value = values.mean()
    std_value = values.std(ddof=1)
    min_value = values.min()
    max_value = values.max()

    # 95% confidence interval
    n = len(values)

    margin = 1.96 * std_value / np.sqrt(n)

    lower = mean_value - margin
    upper = mean_value + margin

    # Συντελεστής μεταβλητότητας
    cv = std_value / mean_value * 100

    statistics.append([
        variable,
        mean_value,
        std_value,
        min_value,
        max_value,
        lower,
        upper,
        cv
    ])


statistics_df = pd.DataFrame(
    statistics,
    columns=[
        "Variable",
        "Mean",
        "Std_Deviation",
        "Minimum",
        "Maximum",
        "CI95_Lower",
        "CI95_Upper",
        "CV_percent"
    ]
)

print("\n\nΠΕΡΙΓΡΑΦΙΚΗ ΣΤΑΤΙΣΤΙΚΗ")
print(statistics_df)

# MIN / MAX ΓΙΑ ΤΙΣ ΜΕΤΑΒΛΗΤΕΣ M1 - M7


print("\n" + "=" * 90)
print("MINIMUM / MAXIMUM ΤΙΜΕΣ M1 - M7")
print("=" * 90)

for variable in columns[1:]:

    min_value = simulation_df[variable].min()
    max_value = simulation_df[variable].max()

    print(f"{variable}:")
    print(f"   Minimum = {min_value:.6f}")
    print(f"   Maximum = {max_value:.6f}")


# ΠΙΝΑΚΑΣ MIN / MAX M1 - M7


min_max_df = pd.DataFrame({
    "Variable": columns[1:],
    "Minimum": [
        simulation_df[var].min()
        for var in columns[1:]
    ],
    "Maximum": [
        simulation_df[var].max()
        for var in columns[1:]
    ]
})

print("\n" + "=" * 90)
print("ΠΙΝΑΚΑΣ MIN / MAX ΜΕΤΑΒΛΗΤΩΝ")
print("=" * 90)

print(min_max_df.to_string(index=False))

min_max_df.to_csv(
    "15_Min_Max_M1_M7.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. ΣΥΓΚΡΙΣΗ ΑΡΧΙΚΩΝ ΚΑΙ MONTE CARLO ΤΙΜΩΝ


comparison_df = df.copy()

for variable in columns[1:]:

    comparison_df[variable + "_MC"] = mean_df[variable]

    comparison_df[variable + "_Change_%"] = (
        (mean_df[variable] - df[variable])
        / df[variable]
    ) * 100

print("\n\nΣΥΓΚΡΙΣΗ ΑΡΧΙΚΩΝ ΚΑΙ MONTE CARLO")
print(comparison_df)



# 5. DEA – ΕΠΙΛΟΓΗ INPUTS / OUTPUTS


# INPUTS:
# M5 = εργαζόμενοι
# M6 = λειτουργικό κόστος
# M7 = αναμεταδόσεις

inputs = ["M5", "M6", "M7"]

# OUTPUTS:
# M1 = κάλυψη
# M2 = διάρκεια ακρόασης
# M3 = μερίδιο ακροαματικότητας
# M4 = διαφημιστικά έσοδα

outputs = ["M1", "M2", "M3", "M4"]



# 6. ΠΙΝΑΚΕΣ X ΚΑΙ Y



X = mean_df[inputs].values.T
Y = mean_df[outputs].values.T

number_of_stations = len(mean_df)



# 7. BCC / VRS – OUTPUT ORIENTED DEA


def dea_bcc_output(station_index):

    """
    BCC / VRS output-oriented DEA.

    Maximize:
        phi

    Subject to:

        X*lambda <= X0

        Y*lambda >= phi*Y0

        sum(lambda) = 1

        lambda >= 0

    Efficiency = 1 / phi
    """

    number_lambda = number_of_stations

    objective = np.zeros(number_lambda + 1)

    # linprog κάνει minimization.

    objective[-1] = -1


    # Inequality constraints

    A_ub = []
    b_ub = []


    # INPUT CONSTRAINTS
    # X lambda <= X0

    for i in range(X.shape[0]):

        constraint = np.zeros(number_lambda + 1)

        constraint[:number_lambda] = X[i]

        A_ub.append(constraint)

        b_ub.append(X[i, station_index])


    # OUTPUT CONSTRAINTS
    # Y lambda >= phi*Y0
    # => -Y lambda + phi*Y0 <= 0

    for r in range(Y.shape[0]):

        constraint = np.zeros(number_lambda + 1)

        constraint[:number_lambda] = -Y[r]

        constraint[-1] = Y[r, station_index]

        A_ub.append(constraint)

        b_ub.append(0)


    # VRS constraint
    # sum(lambda) = 1

    A_eq = np.zeros((1, number_lambda + 1))

    A_eq[0, :number_lambda] = 1

    b_eq = [1]


    # Bounds

    bounds = []

    for i in range(number_lambda):
        bounds.append((0, None))

    # phi >= 1
    bounds.append((1, None))


    # Επίλυση γραμμικού προγράμματος

    result = linprog(
        objective,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )


    if not result.success:

        raise RuntimeError(
            "DEA failed for "
            + mean_df.loc[station_index, "Station"]
        )


    # Αποτελέσματα

    lambda_values = result.x[:number_lambda]

    phi = result.x[-1]

    efficiency = 1 / phi


    # Projection στο αποδοτικό σύνορο


    X_target = X @ lambda_values

    Y_target = Y @ lambda_values


    # INPUT SLACK


    input_slack = (
        X[:, station_index]
        - X_target
    )


    # OUTPUT SLACK

    output_slack = (
        Y_target
        - phi * Y[:, station_index]
    )


    # Reference stations

    references = []

    for j in range(number_lambda):

        if lambda_values[j] > 1e-7:

            references.append(
                (
                    mean_df.loc[j, "Station"],
                    lambda_values[j]
                )
            )


    return {
        "lambda": lambda_values,
        "phi": phi,
        "efficiency": efficiency,
        "X_target": X_target,
        "Y_target": Y_target,
        "input_slack": input_slack,
        "output_slack": output_slack,
        "references": references
    }


# 8. ΕΦΑΡΜΟΓΗ DEA ΣΕ ΟΛΟΥΣ ΤΟΥΣ ΣΤΑΘΜΟΥΣ


solutions = []

for i in range(number_of_stations):

    solution = dea_bcc_output(i)

    solutions.append(solution)



# 9. ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ DEA


dea_results = []

for i, solution in enumerate(solutions):

    efficiency = solution["efficiency"]

    # Έλεγχος slacks
    input_slack_positive = np.any(
        solution["input_slack"] > 1e-6
    )

    output_slack_positive = np.any(
        solution["output_slack"] > 1e-6
    )

    has_slack = (
            input_slack_positive
            or output_slack_positive
    )


    # Χαρακτηρισμός


    if efficiency >= 1 - 1e-7:

        status = "Αποδοτικός"

    elif has_slack:

        status = "Μη αποδοτικός - μικτή μη-αποδοτικότητα"

    else:

        status = "Μη αποδοτικός - ακτινική μη-αποδοτικότητα"




    # Reference set

    reference_text = ""

    for station, lambda_value in solution["references"]:

        reference_text += (
            f"{station} "
            f"(λ={lambda_value:.6f}); "
        )


    dea_results.append([
        mean_df.loc[i, "Station"],
        efficiency,
        solution["phi"],
        input_slack_positive,
        output_slack_positive,
        status,
        reference_text
    ])


dea_results_df = pd.DataFrame(
    dea_results,
    columns=[
        "Station",
        "Efficiency",
        "Phi",
        "Input_slack_positive",
        "Output_slack_positive",
        "Status",
        "Reference_Stations"
    ]
)



# 10. ΕΜΦΑΝΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ DEA


print("\n\n")
print("=" * 90)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΑΔ")
print("=" * 90)

print(
    dea_results_df.to_string(index=False)
)

# ============== MIN MAX EFFICIENCY====================


min_efficiency = dea_results_df["Efficiency"].min()
max_efficiency = dea_results_df["Efficiency"].max()

min_eff_station = dea_results_df.loc[
    dea_results_df["Efficiency"].idxmin(), "Station"
]

max_eff_stations = dea_results_df.loc[
    dea_results_df["Efficiency"] >= 1 - 1e-7,
    "Station"
].tolist()

print("\n" + "=" * 90)
print("MIN / MAX DEA EFFICIENCY")
print("=" * 90)

print(f"Minimum Efficiency: {min_efficiency:.6f}")
print(f"Minimum Efficiency (%): {min_efficiency * 100:.2f}%")
print(f"Station with minimum efficiency: {min_eff_station}")

print(f"\nMaximum Efficiency: {max_efficiency:.6f}")
print(f"Maximum Efficiency (%): {max_efficiency * 100:.2f}%")
print("Efficient stations:")
for station in max_eff_stations:
    print(" -", station)



# 11. ΠΙΝΑΚΑΣ λ


lambda_matrix = np.zeros(
    (number_of_stations, number_of_stations)
)

for i, solution in enumerate(solutions):

    lambda_matrix[i, :] = solution["lambda"]


lambda_df = pd.DataFrame(
    lambda_matrix,
    index=mean_df["Station"],
    columns=mean_df["Station"]
)

print("\n\n")
print("=" * 90)
print("ΠΙΝΑΚΑΣ λ")
print("=" * 90)

print(lambda_df)



# 12. SLACKS


slack_results = []

for i, solution in enumerate(solutions):

    row = {
        "Station":
            mean_df.loc[i, "Station"]
    }


    # Input slacks
    for j, variable in enumerate(inputs):

        row[variable + "_slack"] = (
            solution["input_slack"][j]
        )


    # Output slacks
    for j, variable in enumerate(outputs):

        row[variable + "_slack"] = (
            solution["output_slack"][j]
        )


    slack_results.append(row)


slacks_df = pd.DataFrame(slack_results)

print("\n\n")
print("=" * 90)
print("SLACKS")
print("=" * 90)

print(slacks_df)



# 13. TARGETS ΕΙΣΡΟΩΝ ΚΑΙ ΕΚΡΟΩΝ


target_results = []

for i, solution in enumerate(solutions):

    row = {
        "Station":
            mean_df.loc[i, "Station"],

        "Efficiency":
            solution["efficiency"]
    }


    # Input targets

    for j, variable in enumerate(inputs):

        row[variable + "_target"] = (
            solution["X_target"][j]
        )



    # Output targets

    for j, variable in enumerate(outputs):

        row[variable + "_target"] = (
            solution["Y_target"][j]
        )


    target_results.append(row)


targets_df = pd.DataFrame(target_results)


print("\n\n")
print("=" * 90)
print("TARGETS")
print("=" * 90)

print(targets_df)



# 14. ΜΟΝΟ ΟΙ ΜΗ ΑΠΟΔΟΤΙΚΟΙ ΣΤΑΘΜΟΙ


inefficient_df = dea_results_df[
    dea_results_df["Efficiency"] < 1 - 1e-7
].copy()


print("\n\n")
print("=" * 90)
print("ΜΗ ΑΠΟΔΟΤΙΚΟΙ ΣΤΑΘΜΟΙ")
print("=" * 90)

print(inefficient_df)



# 15. TARGETS ΜΟΝΟ ΓΙΑ ΜΗ ΑΠΟΔΟΤΙΚΟΥΣ


inefficient_names = inefficient_df["Station"].tolist()

inefficient_targets_df = targets_df[
    targets_df["Station"].isin(inefficient_names)
].copy()


print("\n\n")
print("=" * 90)
print("TARGETS ΜΗ ΑΠΟΔΟΤΙΚΩΝ ΣΤΑΘΜΩΝ")
print("=" * 90)

print(inefficient_targets_df)


# 17. ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΕ CSV


print("\n")
print("=" * 90)
print("ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΣΕ CSV")
print("=" * 90)


# ΕΡΩΤΗΣΗ 1
# Monte Carlo - Περιγραφική στατιστική

statistics_df.to_csv(
    "01_MC_Statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

# Σύγκριση αρχικών τιμών με Monte Carlo μέσους
comparison_df.to_csv(
    "02_Base_vs_MonteCarlo.csv",
    index=False,
    encoding="utf-8-sig"
)

# Μέσες τιμές Monte Carlo ανά σταθμό
mean_df.to_csv(
    "03_MC_Means.csv",
    index=False,
    encoding="utf-8-sig"
)


# ΕΡΩΤΗΣΗ 2
# Inputs / Outputs

inputs_outputs_df = pd.DataFrame({
    "Type": ["Input", "Input", "Input",
             "Output", "Output", "Output", "Output"],
    "Variable": [
        "M5", "M6", "M7",
        "M1", "M2", "M3", "M4"
    ],
    "Description": [
        "Αριθμός εργαζομένων",
        "Μηνιαίο λειτουργικό κόστος",
        "Αναμεταδόσεις",
        "Μέση ημερήσια κάλυψη",
        "Μέση διάρκεια ακρόασης",
        "Μερίδιο ακροαματικότητας",
        "Έσοδα από διαφημίσεις"
    ]
})

inputs_outputs_df.to_csv(
    "04_Inputs_Outputs.csv",
    index=False,
    encoding="utf-8-sig"
)

# ΕΡΩΤΗΣΗ 3
# Μοντέλο DEA

model_df = pd.DataFrame({
    "Parameter": [
        "DEA Model",
        "Returns to Scale",
        "Orientation",
        "Inputs",
        "Outputs"
    ],
    "Value": [
        "BCC",
        "Variable Returns to Scale (VRS)",
        "Output Oriented",
        ", ".join(inputs),
        ", ".join(outputs)
    ]
})

model_df.to_csv(
    "05_DEA_Model.csv",
    index=False,
    encoding="utf-8-sig"
)


# ΕΡΩΤΗΣΗ 4
# Αποδοτικότητα, Phi, Status, References

dea_results_df.to_csv(
    "06_DEA_Results.csv",
    index=False,
    encoding="utf-8-sig"
)

# Πίνακας λ
lambda_df.to_csv(
    "07_Lambda.csv",
    index=True,
    encoding="utf-8-sig"
)

# Slacks
slacks_df.to_csv(
    "08_Slacks.csv",
    index=False,
    encoding="utf-8-sig"
)


# ΕΡΩΤΗΣΗ 5
# Targets όλων των σταθμών

targets_df.to_csv(
    "09_Targets_All_Stations.csv",
    index=False,
    encoding="utf-8-sig"
)


# Targets μόνο μη αποδοτικών
inefficient_targets_df.to_csv(
    "10_Targets_Inefficient_Stations.csv",
    index=False,
    encoding="utf-8-sig"
)


# Μη αποδοτικοί σταθμοί

inefficient_df.to_csv(
    "11_Inefficient_Stations.csv",
    index=False,
    encoding="utf-8-sig"
)


# Slacks μόνο μη αποδοτικών

inefficient_slacks_df = slacks_df[
    slacks_df["Station"].isin(inefficient_names)
].copy()

inefficient_slacks_df.to_csv(
    "12_Slacks_Inefficient_Stations.csv",
    index=False,
    encoding="utf-8-sig"
)


# 18. ΣΥΝΟΠΤΙΚΟΣ ΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ

efficient_count = sum(
    dea_results_df["Efficiency"] >= 1 - 1e-7
)

inefficient_count = sum(
    dea_results_df["Efficiency"] < 1 - 1e-7
)

# Μεταβλητή με το μεγαλύτερο CV
most_variable_row = statistics_df.loc[
    statistics_df["CV_percent"].idxmax()
]

summary_df = pd.DataFrame({
    "Item": [
        "Αριθμός σταθμών",
        "Monte Carlo παρατηρήσεις ανά σταθμό",
        "Συνολικές Monte Carlo παρατηρήσεις",
        "DEA μοντέλο",
        "Κλίμακα αποδόσεων",
        "Προσανατολισμός",
        "Inputs",
        "Outputs",
        "Μεταβλητή μεγαλύτερης μεταβλητότητας",
        "Μέγιστο CV (%)",
        "Αριθμός αποδοτικών σταθμών",
        "Αριθμός μη αποδοτικών σταθμών"
    ],
    "Result": [
        len(mean_df),
        N,
        len(simulation_df),
        "BCC",
        "VRS",
        "Output Oriented",
        ", ".join(inputs),
        ", ".join(outputs),
        most_variable_row["Variable"],
        most_variable_row["CV_percent"],
        efficient_count,
        inefficient_count
    ]
})

summary_df.to_csv(
    "13_Final_Summary.csv",
    index=False,
    encoding="utf-8-sig"
)



# 19. ΠΙΝΑΚΑΣ ΓΙΑ ΤΗΝ ΤΕΛΙΚΗ ΠΑΡΟΥΣΙΑΣΗ


final_dea_df = dea_results_df[
    [
        "Station",
        "Efficiency",
        "Phi",
        "Status",
        "Reference_Stations"
    ]
].copy()

final_dea_df["Efficiency_%"] = (
    final_dea_df["Efficiency"] * 100
)

final_dea_df.to_csv(
    "14_Final_DEA_Table.csv",
    index=False,
    encoding="utf-8-sig"
)


# 20. ΕΛΕΓΧΟΣ ΑΡΧΕΙΩΝ


print("\nΤα παρακάτω CSV δημιουργήθηκαν στον ίδιο φάκελο")
print("όπου εκτελείται το Python πρόγραμμα:\n")

csv_files = [
    "01_MC_Statistics.csv",
    "02_Base_vs_MonteCarlo.csv",
    "03_MC_Means.csv",
    "04_Inputs_Outputs.csv",
    "05_DEA_Model.csv",
    "06_DEA_Results.csv",
    "07_Lambda.csv",
    "08_Slacks.csv",
    "09_Targets_All_Stations.csv",
    "10_Targets_Inefficient_Stations.csv",
    "11_Inefficient_Stations.csv",
    "12_Slacks_Inefficient_Stations.csv",
    "13_Final_Summary.csv",
    "14_Final_DEA_Table.csv"
]

for file in csv_files:
    print("✓", file)

print("\n")
print("=" * 90)
print("Η ΔΗΜΙΟΥΡΓΙΑ ΤΩΝ CSV ΟΛΟΚΛΗΡΩΘΗΚΕ")
print("=" * 90)

print("\n\n")
print("=" * 90)
print("ΟΛΟΚΛΗΡΩΘΗΚΕ Η ΥΛΟΠΟΙΗΣΗ")
print("=" * 90)

print(
    "\nΑποδοτικοί σταθμοί:",
    sum(dea_results_df["Efficiency"] >= 1 - 1e-7)
)

print(
    "Μη αποδοτικοί σταθμοί:",
    sum(dea_results_df["Efficiency"] < 1 - 1e-7)
)
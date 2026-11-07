import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 1000

sucursales = ["El Progreso", "Santa Ana", "San Miguel", "29 Calle Pte", "Venezuela", "Roosevelt",
              "Soyapango", "Plaza Mundo", "Zacatecoluca", "Usulutan", "Sonsonate", "Apopa",
              "Aguilares", "Lourdes", "Gotera", "Ahuachapan", "Cara Sucia", "Chalatenango",
              "San Miguel Centro", "Acajutla", "Metapan", "Santa Tecla", "La Union", "Ruta Militar San Miguel"]
peso_suc = np.array([10, 8, 8, 6, 5, 5, 7, 6, 4, 4, 5, 4, 3, 3, 2, 3, 2, 3, 3, 2, 2, 4, 2, 2], float)
peso_suc /= peso_suc.sum()

# (marca, modelo, cilindraje, precio)
modelos = [
    ("KATANA", "MK1-MT", 125, 1250), ("KATANA", "MK1-CVT", 125, 1350), ("KATANA", "NAVE 150", 150, 1450),
    ("KATANA", "KN 150", 150, 1500), ("KATANA", "NITRO 150", 150, 1650), ("KATANA", "SMX 150", 150, 1550),
    ("KATANA", "CRII 150", 150, 1400), ("KATANA", "SMX 200", 200, 2000), ("KATANA", "R1-200", 200, 2100),
    ("KATANA", "TERRA 200", 200, 2250), ("KATANA", "SMX 250", 250, 2500), ("KATANA", "R1-250", 250, 2700),
    ("KATANA", "R2-250", 250, 2800), ("KATANA", "KR-250", 250, 2950), ("KATANA", "CK250-2R", 250, 2600),
    ("KATANA", "TERRA 330", 330, 3450), ("DAYUN", "125-6", 125, 1200), ("DAYUN", "150-6E", 150, 1350),
    ("DAYUN", "150-16D", 150, 1450), ("DAYUN", "150-22A", 150, 1550), ("DAYUN", "150-29A", 150, 1600),
    ("DAYUN", "200 GY", 200, 2050), ("DAYUN", "200H", 200, 1900), ("DAYUN", "MECHA 200", 200, 2300),
    ("DAYUN", "250-R", 250, 2650), ("DAYUN", "WORKING 250", 250, 2900), ("KATANA", "SCRAMBLER 250", 250, 2750),
]
peso_mod = np.array([6 if m[2] <= 150 else 3 if m[2] == 200 else 1.6 for m in modelos], float)
peso_mod /= peso_mod.sum()

idx_mod = rng.choice(len(modelos), N, p=peso_mod)
marca = [modelos[i][0] for i in idx_mod]
modelo = [modelos[i][1] for i in idx_mod]
cc = np.array([modelos[i][2] for i in idx_mod])
precio = np.array([modelos[i][3] for i in idx_mod], float)
precio = precio + rng.choice([-50, 0, 0, 50], N)

tipo_ingreso = rng.choice(["Empleo formal", "Negocio propio", "Remesas"], N, p=[0.55, 0.30, 0.15])
ingreso = np.where(tipo_ingreso == "Empleo formal", rng.normal(560, 170, N),
          np.where(tipo_ingreso == "Negocio propio", rng.normal(620, 260, N), rng.normal(430, 120, N)))
ingreso = np.clip(ingreso, 300, 1800).round(0)

edad = np.clip(rng.normal(33, 9, N), 18, 65).round(0).astype(int)
anios_trabajo = np.clip(rng.gamma(2.0, 2.0, N), 0, 30).round(1)
dependientes = rng.choice([0, 1, 2, 3, 4, 5], N, p=[0.2, 0.25, 0.25, 0.17, 0.09, 0.04])
comprobantes = np.where(tipo_ingreso == "Empleo formal", rng.choice(["Si", "No"], N, p=[0.92, 0.08]),
               rng.choice(["Si", "No"], N, p=[0.6, 0.4]))
credito_previo = rng.choice(["Si", "No"], N, p=[0.55, 0.45])

prima_pct = np.clip(rng.normal(0.22, 0.07, N), 0.15, 0.50)
prima = (precio * prima_pct).round(2)
plazo = rng.choice([12, 18, 24, 36, 48], N, p=[0.10, 0.12, 0.30, 0.30, 0.18])
financiado = precio - prima
tasa_mensual = 0.025
cuota = (financiado * tasa_mensual / (1 - (1 + tasa_mensual) ** (-plazo))).round(2)

fechas = pd.to_datetime("2023-01-02") + pd.to_timedelta(rng.integers(0, 900, N), unit="D")

# --- Probabilidad de mora (lógica de negocio) ---
carga = cuota / ingreso
logit = (-1.6
         + 14.0 * (carga - 0.20)
         - 10.0 * (prima_pct - 0.22)
         + 0.045 * (plazo - 24)
         + np.where(tipo_ingreso == "Remesas", 1.0, np.where(tipo_ingreso == "Negocio propio", 0.55, 0))
         + np.where(comprobantes == "No", 1.3, 0)
         - np.where(credito_previo == "Si", 0.9, 0)
         + np.where(edad < 23, 0.8, 0)
         + np.where(anios_trabajo < 1, 0.9, 0)
         + 0.2 * (dependientes - 2)
         + rng.normal(0, 0.3, N))
prob = 1 / (1 + np.exp(-logit))
mora = (rng.random(N) < prob).astype(int)

df = pd.DataFrame({
    "DocEntry": np.arange(10501, 10501 + N),
    "CardCode": [f"C{c:05d}" for c in rng.choice(np.arange(1000, 40000), N, replace=False)],
    "DocDate": fechas.strftime("%Y-%m-%d"),
    "U_Sucursal": rng.choice(sucursales, N, p=peso_suc),
    "U_Marca": marca,
    "U_Modelo": modelo,
    "U_Cilindraje": cc,
    "U_PrecioMoto": precio,
    "U_Prima": prima,
    "U_Plazo": plazo,
    "U_Cuota": cuota,
    "U_Edad": edad,
    "U_TipoIngreso": tipo_ingreso,
    "U_IngresoMensual": ingreso,
    "U_AniosTrabajo": anios_trabajo,
    "U_Dependientes": dependientes,
    "U_Comprobantes": comprobantes,
    "U_CreditoPrevio": credito_previo,
    "U_Mora": mora,
})
df = df.sort_values("DocDate").reset_index(drop=True)
df["DocEntry"] = np.arange(10501, 10501 + N)

# --- Errores a propósito (para la sección de limpieza) ---
n = len(df)
i_null_ing = rng.choice(n, 18, replace=False)
df.loc[i_null_ing, "U_IngresoMensual"] = np.nan
i_null_anios = rng.choice(n, 11, replace=False)
df.loc[i_null_anios, "U_AniosTrabajo"] = np.nan
i_null_suc = rng.choice(n, 6, replace=False)
df.loc[i_null_suc, "U_Sucursal"] = np.nan

# edades imposibles
i_edad = rng.choice(n, 5, replace=False)
df.loc[i_edad, "U_Edad"] = [3, 150, 0, 210, 9]
# ingresos absurdos (error de digitación: un cero de más)
i_ing = rng.choice([i for i in range(n) if i not in i_null_ing], 4, replace=False)
df.loc[i_ing, "U_IngresoMensual"] = df.loc[i_ing, "U_IngresoMensual"] * 100
# prima negativa
i_prima = rng.choice(n, 3, replace=False)
df.loc[i_prima, "U_Prima"] = -df.loc[i_prima, "U_Prima"]
# sucursales mal escritas
i_suc = rng.choice([i for i in range(n) if i not in i_null_suc], 14, replace=False)
variantes = {"Santa Ana": "santa ana", "San Miguel": "SAN MIGUEL ", "El Progreso": "El progreso",
             "Soyapango": "Soyapango ", "Santa Tecla": "Sta. Tecla", "Sonsonate": "SONSONATE"}
for i in i_suc:
    s = df.at[i, "U_Sucursal"]
    df.at[i, "U_Sucursal"] = variantes.get(s, s.upper())
# marca en minúscula en algunos
i_marca = rng.choice(n, 8, replace=False)
df.loc[i_marca, "U_Marca"] = df.loc[i_marca, "U_Marca"].str.capitalize()

# duplicados exactos (registro guardado dos veces)
dups = df.sample(12, random_state=7)
df = pd.concat([df, dups]).sort_values("DocEntry").reset_index(drop=True)

df.to_csv("creditos_motocity.csv", index=False)
print(df.shape)
print(df["U_Mora"].value_counts(normalize=True))

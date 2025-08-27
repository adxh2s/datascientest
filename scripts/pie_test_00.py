import numpy as np
import matplotlib.pyplot as plt

# --- 1. Vos données ---
labels = ['A', 'B', 'C', 'D']
sizes = [50, 25, 15, 10]  # exemple : les pourcentages (somme = 100)

# --- 2. Construction du camembert ---
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(aspect="equal"))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct='%1.0f%%', startangle=90
)

# --- 3. Trouver les angles de chacun des secteurs ---
angles = np.cumsum([0] + sizes) * 360 / sum(sizes)
# angles[i] = angle de début du secteur i, en degrés

# --- 4. Détecter secteur le plus grand et le plus petit ---
i_max = np.argmax(sizes)
i_min = np.argmin(sizes)

angle_start = angles[i_max]    # début du secteur le plus grand
angle_end = angles[i_min+1]    # fin du plus petit secteur

# Si les secteurs sont dans le désordre, vous pouvez trier vos données
# et réassocier labels/colors si besoin

# --- 5. Tracer une flèche tournante avec graduation ---
# Passage en radians
theta1 = np.deg2rad(angle_start)
theta2 = np.deg2rad(angle_end)
Ngradu = 10
gradus = np.linspace(theta1, theta2, Ngradu)
cumul_sizes = np.cumsum([sizes[i_max]] + sizes[i_max+1:i_min+1])
# Tracer l'arc/flèche
r = 1.15  # Rayon juste autour du camembert

ax.plot(
    r*np.cos(gradus), r*np.sin(gradus), color='red', lw=2, zorder=10
)

# Ajout d'une tête de flèche à la fin
arrow_x = r*np.cos(theta2)
arrow_y = r*np.sin(theta2)
ax.annotate(
    '', xy=(arrow_x, arrow_y), xytext=(r*np.cos(theta2-0.05), r*np.sin(theta2-0.05)),
    arrowprops=dict(facecolor='red', shrink=0.05, width=5, headwidth=20)
)

# Ajout de graduations cumulative en %
for idx, g in enumerate(gradus):
    pct = int((angles[0] + (g-theta1)/(theta2-theta1)*(angle_end-angle_start)))
    ax.text(
        (r+0.07)*np.cos(g), (r+0.07)*np.sin(g), f"{pct}%", 
        va='center', ha='center', fontsize=8, color='red', rotation=np.rad2deg(g)+90
    )

plt.title("Camembert avec flèche tournante et graduation cumulative")
plt.tight_layout()
plt.show()

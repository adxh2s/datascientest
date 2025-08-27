import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.ion()  # Mode interactif matplotlib activé


def recherche_bas_haut(X_scaled, Z, nb_top=3, sil_min=0.51, sil_max=1.0, profondeur=20):
    heights = Z[:, 2]
    diffs = np.diff(heights[::-1])
    top_jump_indices = np.argsort(diffs)[-profondeur:]
    candidate_heights = heights[::-1][top_jump_indices + 1]

    scores = []
    for h in candidate_heights:
        labels = fcluster(Z, t=h, criterion='distance')
        n_clust = len(np.unique(labels))
        if n_clust < 2 or n_clust >= len(X_scaled):
            continue
        score = silhouette_score(X_scaled, labels)
        if sil_min <= score <= sil_max:
            scores.append((h, score, n_clust))

    if len(scores) < nb_top:
        distances = []
        for h in candidate_heights:
            labels = fcluster(Z, t=h, criterion='distance')
            n_clust = len(np.unique(labels))
            if n_clust < 2 or n_clust >= len(X_scaled):
                continue
            score = silhouette_score(X_scaled, labels)
            distances.append((h, score, n_clust, abs(score - sil_min)))
        distances.sort(key=lambda x: x[3])
        for entry in distances:
            if entry[:3] not in [(x[0], x[1], x[2]) for x in scores]:
                scores.append(entry[:3])
            if len(scores) >= nb_top:
                break

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:nb_top]


def recherche_haut_bas(X_scaled, Z, nb_top=3, min_clusters=3, profondeur=100):
    max_height = np.max(Z[:, 2])
    possible_heights = np.linspace(max_height * 0.9, 0, profondeur)
    heights_clusters = []
    last_clusters = -1
    for h in possible_heights:
        labels = fcluster(Z, t=h, criterion='distance')
        n_clusters = len(np.unique(labels))
        if n_clusters >= min_clusters and n_clusters != last_clusters:
            heights_clusters.append((h, n_clusters))
            last_clusters = n_clusters
            if len(heights_clusters) == nb_top:
                break

    scores = []
    for h, n_c in heights_clusters:
        labels = fcluster(Z, t=h, criterion='distance')
        score = silhouette_score(X_scaled, labels)
        scores.append((h, score, n_c))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def save_figure(fig, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"[INFO] Image sauvegardée sous {filepath}")


def plot_scores(scores_list, title_prefix, X_scaled, Z, save_dir=None, direction="B_U"):
    n = len(scores_list)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = np.array([axes])

    for i, (h, s, n_c) in enumerate(scores_list):
        labels = fcluster(Z, t=h, criterion='distance')

        ax_scatter = axes[i, 0]
        sc = ax_scatter.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=30)
        ax_scatter.set_title(f'{title_prefix} Hauteur {h:.3f} - {n_c} clusters')
        ax_scatter.set_xlabel('Feature 1')
        ax_scatter.set_ylabel('Feature 2')
        fig.colorbar(sc, ax=ax_scatter)

        ax_bar = axes[i, 1]
        bars = ax_bar.bar([0], [s], color='skyblue')
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xticks([0])
        ax_bar.set_xticklabels(['Silhouette'])
        ax_bar.set_title('Score silhouette moyen')
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.3f}',
                       ha='center', va='bottom')

        ax_dendro = axes[i, 2]
        dendrogram(Z, truncate_mode='level', p=10, ax=ax_dendro, color_threshold=h)
        ax_dendro.axhline(y=h, color='red', linestyle='--')
        ax_dendro.set_title(f'{title_prefix} Dendrogramme coupé à {h:.3f}')
        ax_dendro.set_xlabel('Observations')
        ax_dendro.set_ylabel('Distance')

    plt.tight_layout()
    fig.show()

    if save_dir:
        for i, (h, s, n_c) in enumerate(scores_list):
            fig_part = plt.figure(figsize=(15, 5))
            ax1 = fig_part.add_subplot(1, 3, 1)
            labels = fcluster(Z, t=h, criterion='distance')
            sc = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', s=30)
            ax1.set_title(f'{title_prefix} Hauteur {h:.3f} - {n_c} clusters')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            fig_part.colorbar(sc, ax=ax1)

            ax2 = fig_part.add_subplot(1, 3, 2)
            bars = ax2.bar([0], [s], color='skyblue')
            ax2.set_ylim(0, 1)
            ax2.set_xticks([0])
            ax2.set_xticklabels(['Silhouette'])
            ax2.set_title('Score silhouette moyen')
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.3f}',
                         ha='center', va='bottom')

            ax3 = fig_part.add_subplot(1, 3, 3)
            dendrogram(Z, truncate_mode='level', p=10, ax=ax3, color_threshold=h)
            ax3.axhline(y=h, color='red', linestyle='--')
            ax3.set_title(f'{title_prefix} Dendrogramme coupé à {h:.3f}')
            ax3.set_xlabel('Observations')
            ax3.set_ylabel('Distance')

            # Nom fichier : Direction_graphiqueNuméroCluster.png ex: B_U_Graphique_5.png
            base_label = direction.upper()
            fname = f"{base_label}_Graphique_{n_c}.png"
            full_path = os.path.join(save_dir, fname)
            save_figure(fig_part, full_path)
            plt.close(fig_part)


def plot_silhouette_curve_with_lines(X_scaled, Z, cluster_marks, save_dir=None):
    heights = np.linspace(np.max(Z[:, 2]) * 0.9, 0, 300)
    silhouette_scores = []
    cluster_counts = []

    for h in heights:
        labels = fcluster(Z, t=h, criterion='distance')
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= len(X_scaled):
            silhouette_scores.append(np.nan)
            cluster_counts.append(n_clusters)
            continue
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        cluster_counts.append(n_clusters)

    silhouette_scores = np.array(silhouette_scores)
    cluster_counts = np.array(cluster_counts)
    valid = ~np.isnan(silhouette_scores)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(cluster_counts[valid], silhouette_scores[valid], marker='o', label='Score silhouette')

    max_idx = np.nanargmax(silhouette_scores)
    min_idx = np.nanargmin(silhouette_scores[valid])
    ax.plot(cluster_counts[max_idx], silhouette_scores[max_idx], 'go', label='Maximum silhouette', markersize=10)
    ax.plot(cluster_counts[valid][min_idx], silhouette_scores[valid][min_idx], 'ro', label='Minimum silhouette', markersize=10)

    colors = plt.cm.get_cmap('tab10').colors
    linestyles = ['--', '-.', ':', (0, (5, 1)), (0, (3, 5, 1, 5))]
    n_styles = len(linestyles)

    legend_lines = []
    legend_labels = []

    for i, c in enumerate(cluster_marks):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % n_styles]
        ax.axvline(x=c, color=color, linestyle=linestyle, alpha=0.8)
        legend_lines.append(mlines.Line2D([], [], color=color, linestyle=linestyle))
        legend_labels.append(f'Graphique {i + 1} (N° cluster: {c})')

    handles, labels = ax.get_legend_handles_labels()
    by_label = {label: handle for label, handle in zip(labels, handles)}
    for label, handle in zip(legend_labels, legend_lines):
        by_label[label] = handle

    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Score silhouette moyen')
    ax.set_title('Évolution du score silhouette selon nombre de clusters')
    ax.grid(True)

    fig.show()

    if save_dir:
        file_path = os.path.join(save_dir, "courbe_silhouette.png")
        save_figure(fig, file_path)
        plt.close(fig)


# --------- Paramètres ---------
depth_top_down = 3
silhouette_min = 0.51
save_dir = "../data/img/"

# --------- Génération données et linkage ---------
np.random.seed(42)
X = np.vstack([
    np.random.randn(60, 2) + [0, 0],
    np.random.randn(15, 2) + [5, 5],
    np.random.randn(30, 2) + [10, 0],
    np.random.randn(10, 2) + [12, 8]
])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=10)
plt.title("Dendrogramme tronqué (10 derniers niveaux)")
plt.xlabel("Observations")
plt.ylabel("Distance")
plt.show()

# Recherche bottom->up et top->down
top_scores_bottom_up = recherche_bas_haut(X_scaled, Z, nb_top=3, sil_min=silhouette_min)
print(f"\nTop 3 bas → haut (silhouette ≥ {silhouette_min} ou proches) :")
for h, s, n_c in top_scores_bottom_up:
    print(f"Hauteur={h:.3f} - Clusters={n_c} - Silhouette={s:.3f}")

top_scores_top_down = recherche_haut_bas(X_scaled, Z, nb_top=depth_top_down, min_clusters=3)
print(f"\nTop {depth_top_down} haut → bas (min 3 clusters) :")
for h, s, n_c in top_scores_top_down:
    print(f"Hauteur={h:.3f} - Clusters={n_c} - Silhouette={s:.3f}")

# Affichages et sauvegardes des graphiques avec noms formatés
plot_scores(top_scores_bottom_up, "Bas → Haut", X_scaled, Z, save_dir, direction="B_U")
plot_scores(top_scores_top_down, "Haut → Bas", X_scaled, Z, save_dir, direction="U_B")

# Préparation liste clusters pour courbe silhouette
cluster_marks = []
for h, _, _ in top_scores_bottom_up:
    labels = fcluster(Z, t=h, criterion='distance')
    cluster_marks.append(len(np.unique(labels)))
for h, _, _ in top_scores_top_down:
    labels = fcluster(Z, t=h, criterion='distance')
    cluster_marks.append(len(np.unique(labels)))

# Courbe silhouette avancée avec sauvegarde
plot_silhouette_curve_with_lines(X_scaled, Z, sorted(set(cluster_marks)), save_dir)

plt.ioff()
plt.show(block=True)

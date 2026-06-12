"""
generate_polish_figures.py
==========================
Generates the four polish figures:
  Fig 6: Wonham filter confusion matrix
  Fig 7: Runtime scaling (events/sec vs simulation size)
  Fig 8: Full pipeline architecture diagram
  Fig 9: Regime detection accuracy vs vol separation (identifiability study)

Run:
    python generate_polish_figures.py
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, 'pde_solver/src')
sys.path.insert(0, 'hmm_filter/src')
from wonham_filter import WonhamFilter, simulate_regime_switching_prices

os.makedirs('results/plots', exist_ok=True)

BG=('#0F0F0F'); PANEL='#1A1A1A'; GRID='#2A2A2A'; TEXT='#CCCCCC'
SUBTLE='#555555'; GREEN='#1D9E75'; BLUE='#378ADD'; ORANGE='#EF9F27'
RED='#D85A30'; PURPLE='#7F77DD'; TEAL='#5DCAA5'; WHITE='#E8E6E0'

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.tick_params(colors=SUBTLE, labelsize=9)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    if title:  ax.set_title(title, color=TEXT, fontsize=11, pad=6)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Confusion matrix for Wonham classifier
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 6: Confusion matrix...")

s1, s2 = 0.005, 0.020
q12, q21 = 1/300, 1/120
T_sec, dt = 7200, 1.0

# Run 50 independent paths and aggregate
tp=tn=fp=fn = 0
for seed in range(50):
    _, S, true_reg = simulate_regime_switching_prices(
        T_sec, dt, s1, s2, q12, q21, seed=seed)
    filt = WonhamFilter(s1, s2, q12, q21, pi_init=q12/(q12+q21), dt=dt)
    pi   = filt.update_batch(S)
    pred = (pi >= 0.5).astype(int)
    tp  += ((pred==1) & (true_reg==1)).sum()
    tn  += ((pred==0) & (true_reg==0)).sum()
    fp  += ((pred==1) & (true_reg==0)).sum()
    fn  += ((pred==0) & (true_reg==1)).sum()

total  = tp+tn+fp+fn
cm     = np.array([[tn, fp],[fn, tp]])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
accuracy = (tp+tn)/total
precision= tp/(tp+fp) if (tp+fp)>0 else 0
recall   = tp/(tp+fn) if (tp+fn)>0 else 0
f1       = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Figure 6 — Wonham Filter: Confusion Matrix  (50 paths × 7200 steps)',
             color=TEXT, fontsize=12)

# Left: count matrix
ax = axes[0]; ax.set_facecolor(PANEL)
colors_cm = [[GREEN, RED], [RED, GREEN]]
alpha_cm  = [[0.7*tn/total*4, 0.7], [0.7, 0.7*tp/total*4]]
for i in range(2):
    for j in range(2):
        val = cm[i,j]; pct = cm_pct[i,j]
        c   = GREEN if i==j else RED
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                     facecolor=c, alpha=0.25 if i==j else 0.15))
        ax.text(j, i, f'{val:,}', ha='center', va='center',
                color=WHITE, fontsize=16, fontweight='bold')
        ax.text(j, i+0.28, f'({pct:.1f}%)', ha='center', va='center',
                color=SUBTLE, fontsize=9)

ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred: Calm','Pred: Chaotic'], color=TEXT, fontsize=10)
ax.set_yticklabels(['True: Calm','True: Chaotic'], color=TEXT, fontsize=10)
ax.set_title('Confusion Matrix (counts)', color=TEXT, fontsize=11)
for sp in ax.spines.values(): sp.set_color('#333')
ax.tick_params(colors=SUBTLE)

# Right: metrics bar chart
ax2 = axes[1]; ax2.set_facecolor(PANEL)
metrics_vals  = [accuracy, precision, recall, f1]
metrics_names = ['Accuracy', 'Precision\n(chaotic)', 'Recall\n(chaotic)', 'F1\n(chaotic)']
colors_m = [TEAL, BLUE, ORANGE, PURPLE]
bars = ax2.bar(range(4), metrics_vals, color=colors_m, alpha=0.75, width=0.5)
for bar, val in zip(bars, metrics_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.005,
             f'{val:.3f}', ha='center', va='bottom', color=TEXT, fontsize=11,
             fontweight='bold')
ax2.set_xticks(range(4)); ax2.set_xticklabels(metrics_names, color=TEXT, fontsize=9)
ax2.set_ylim(0, 1.12); ax2.axhline(1.0, color=SUBTLE, lw=0.7, ls='--')
style_ax(ax2, title='Classification Metrics', ylabel='Score')

# Summary text
summary = (f"50 paths · {T_sec} steps each · σ₁={s1} · σ₂={s2}\n"
           f"TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
fig.text(0.5, 0.01, summary, ha='center', color=SUBTLE, fontsize=9)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig('results/plots/fig6_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  fig6 saved  accuracy={accuracy:.1%}  F1={f1:.3f}")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Runtime scaling plot
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 7: Runtime scaling...")

sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
kappa=2.0; gamma=0.3; dt_sim=1/60; q_max=20
sigma1_bt=0.5; sigma2_bt=3.0; A1=5.0; A2=50.0; q12b=2.0; q21b=8.0

# Component 1: path generation
times_path = []
for N in sizes:
    runs = []
    for _ in range(4):
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        regime=0; S=100.0
        for i in range(N):
            if regime==0:
                if rng.random()<q12b*dt_sim: regime=1
            else:
                if rng.random()<q21b*dt_sim: regime=0
            sv=sigma2_bt if regime==1 else sigma1_bt
            S+=sv*rng.standard_normal()*np.sqrt(dt_sim)
        runs.append(time.perf_counter()-t0)
    times_path.append(np.median(runs))

# Component 2: full sim step (path + spread + arrivals)
times_full = []
for N in sizes:
    runs = []
    for _ in range(4):
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        regime=0; S=100.0; cash=0.0; q=0
        sv_avg=(q12b/(q12b+q21b))*sigma2_bt+(q21b/(q12b+q21b))*sigma1_bt
        for i in range(N):
            tau=max(1.0-i*dt_sim,1e-6)
            if regime==0:
                if rng.random()<q12b*dt_sim: regime=1
            else:
                if rng.random()<q21b*dt_sim: regime=0
            sv=sigma2_bt if regime==1 else sigma1_bt
            S+=sv*rng.standard_normal()*np.sqrt(dt_sim)
            if abs(q)<q_max:
                da=max(1/kappa+0.5*gamma*sv**2*tau-gamma*sv**2*tau*q,1e-4)
                db=max(1/kappa+0.5*gamma*sv**2*tau+gamma*sv**2*tau*q,1e-4)
                A_k=A2 if regime==1 else A1
                if rng.random() < (1.0 - np.exp(-A_k*np.exp(-kappa*da)*dt_sim)):
                    cash+=S+da; q-=1
                if abs(q)<q_max and rng.random() < (1.0 - np.exp(-A_k*np.exp(-kappa*db)*dt_sim)):
                    cash-=S-db; q+=1
        runs.append(time.perf_counter()-t0)
    times_full.append(np.median(runs))

# Component 3: Wonham filter
times_wonham = []
s1w, s2w = 0.005, 0.020
for N in sizes:
    _, S_path, _ = simulate_regime_switching_prices(N,1.0,s1w,s2w,1/300,1/120,seed=0)
    runs = []
    for _ in range(4):
        filt2 = WonhamFilter(s1w,s2w,1/300,1/120,pi_init=0.2,dt=1.0)
        t0=time.perf_counter(); filt2.update_batch(S_path); runs.append(time.perf_counter()-t0)
    times_wonham.append(np.median(runs))

sizes_arr = np.array(sizes)
rate_path   = sizes_arr / np.array(times_path)   / 1e3  # k steps/sec
rate_full   = sizes_arr / np.array(times_full)   / 1e3
rate_wonham = sizes_arr / np.array(times_wonham) / 1e3

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Figure 7 — Runtime Scaling  (median of 4 runs per size)',
             color=TEXT, fontsize=12)

# Left: wall time
ax = axes[0]; ax.set_facecolor(PANEL)
ax.loglog(sizes, times_path,   'o-', color=TEAL,   lw=2, ms=5, label='Path generation only')
ax.loglog(sizes, times_full,   's-', color=ORANGE, lw=2, ms=5, label='Full sim step')
ax.loglog(sizes, times_wonham, '^-', color=PURPLE, lw=2, ms=5, label='Wonham filter')
# O(n) reference line
ref = np.array(times_full[0]) * sizes_arr / sizes[0]
ax.loglog(sizes, ref, '--', color=SUBTLE, lw=1, alpha=0.7, label='O(n) reference')
style_ax(ax, title='Wall Time vs Simulation Size',
         xlabel='N (steps)', ylabel='Time (seconds)')
ax.legend(fontsize=8, facecolor='#222', labelcolor=TEXT, framealpha=0.8)

# Right: throughput
ax2 = axes[1]; ax2.set_facecolor(PANEL)
ax2.semilogx(sizes, rate_path/1e3,   'o-', color=TEAL,   lw=2, ms=5, label='Path generation only')
ax2.semilogx(sizes, rate_full/1e3,   's-', color=ORANGE, lw=2, ms=5, label='Full sim step')
ax2.semilogx(sizes, rate_wonham/1e3, '^-', color=PURPLE, lw=2, ms=5, label='Wonham filter')
style_ax(ax2, title='Throughput vs Simulation Size',
         xlabel='N (steps)', ylabel='Throughput (M steps/sec)')
ax2.legend(fontsize=8, facecolor='#222', labelcolor=TEXT, framealpha=0.8)

# Annotate final throughput values
for rate, color, label in [
    (rate_full[-1]/1e3,   ORANGE, f'{rate_full[-1]/1e3:.3f}M/s'),
    (rate_wonham[-1]/1e3, PURPLE, f'{rate_wonham[-1]/1e3:.3f}M/s'),
]:
    ax2.annotate(label, (sizes[-1], rate),
                 xytext=(-55, 8), textcoords='offset points',
                 color=color, fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('results/plots/fig7_runtime_scaling.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print(f"  fig7 saved  full={rate_full[-1]/1e3:.3f}M/s  wonham={rate_wonham[-1]/1e3:.3f}M/s")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 8 — Pipeline architecture diagram
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 8: Pipeline diagram...")

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis('off')
ax.set_title('Figure 8 — Regime-Switching Market Making: Full Pipeline',
             color=TEXT, fontsize=13, pad=12)

def box(ax, x, y, w, h, label, sublabel, color, icon=''):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, alpha=0.2,
                          edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y+0.18, f'{icon} {label}' if icon else label,
            ha='center', va='center', color=WHITE,
            fontsize=10, fontweight='bold')
    ax.text(x, y-0.22, sublabel, ha='center', va='center',
            color=SUBTLE, fontsize=8)

def arrow(ax, x1, y1, x2, y2, label='', color=SUBTLE):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.5, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.15, label, ha='center', color=color, fontsize=8)

# Row 1: inputs
box(ax, 2.0, 6.5, 2.8, 1.0, 'Hidden Regime', 'k_t ∈ {calm, chaotic}', RED, '⬡')
box(ax, 6.5, 6.5, 2.8, 1.0, 'Price Process', 'dS_t = σ_{k_t} dW_t', TEAL, '~')
box(ax, 11.0,6.5, 2.8, 1.0, 'Order Flow', 'λ_k(δ) = A_k e^{−κδ}', BLUE, '↕')

# Row 2: inference & control
box(ax, 2.0, 4.5, 2.8, 1.0, 'Wonham Filter', 'dπ_t = drift + gain·dI_t', PURPLE, '🔮')
box(ax, 6.5, 4.5, 2.8, 1.0, 'HJB PDE Solver', 'Crank-Nicolson\n1000t × 41q × 2k', ORANGE, '∂')
box(ax, 11.0,4.5, 2.8, 1.0, 'Spread Table', 'δ*(q, t, k) precomputed\nlookup at runtime', GREEN, '📋')

# Row 3: execution
box(ax, 6.5, 2.5, 2.8, 1.0, 'Simulator', 'Poisson arrivals\nAdverse selection', BLUE, '⚡')

# Row 4: output
box(ax, 2.5, 0.8, 2.8, 0.9, 'PnL Series', 'Terminal wealth\nper simulation', TEAL, '📈')
box(ax, 6.5, 0.8, 2.8, 0.9, 'Risk Metrics', 'Sharpe · CVaR\nInv variance', ORANGE, '📊')
box(ax, 10.5,0.8, 2.8, 0.9, 'Ablation Table', 'Component\ncontribution', GREEN, '🔬')

# Arrows
arrow(ax, 2.0,6.0, 2.0,5.0,  'σ_k drives', RED)
arrow(ax, 6.5,6.0, 6.5,5.0,  'price moves', TEAL)
arrow(ax, 3.4,6.5, 5.1,6.5,  'switches', RED)
arrow(ax, 8.0,6.5, 9.6,6.5,  'rate λ_k', BLUE)
arrow(ax, 2.0,4.0, 4.5,2.9,  'π_t', PURPLE)
arrow(ax, 6.5,5.0, 6.5,3.0,  '', ORANGE)
arrow(ax, 9.6,4.5, 8.0,3.0,  'δ*(q,t,k)', GREEN)
arrow(ax, 9.6,6.5, 9.6,5.0,  'A_k', BLUE)
arrow(ax, 6.5,2.0, 2.5,1.25, '', TEAL)
arrow(ax, 6.5,2.0, 6.5,1.25, '', ORANGE)
arrow(ax, 6.5,2.0,10.5,1.25, '', GREEN)

# Offline / Online labels
ax.text(0.3, 5.5, 'OFFLINE\n(one-time)', color=ORANGE, fontsize=9,
        fontweight='bold', alpha=0.8, rotation=90, va='center')
ax.text(0.3, 3.5, 'ONLINE\n(per step)', color=PURPLE, fontsize=9,
        fontweight='bold', alpha=0.8, rotation=90, va='center')
ax.axhline(4.0, color='#333', lw=0.8, ls='--', xmin=0.06, xmax=0.94)

plt.tight_layout()
plt.savefig('results/plots/fig8_pipeline_diagram.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close(); print("  fig8 saved")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 9 — Detection accuracy vs vol separation (identifiability)
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 9: Identifiability study...")

sigma_base = 0.005
ratios     = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
n_paths    = 30
T_id, dt_id = 7200, 1.0

accs_mean = []; accs_std = []
prec_mean = []; rec_mean = []

for ratio in ratios:
    s2 = sigma_base * ratio
    path_accs = []; path_prec = []; path_rec = []
    for seed in range(n_paths):
        _, S, true_reg = simulate_regime_switching_prices(
            T_id, dt_id, sigma_base, s2, q12, q21, seed=seed)
        filt = WonhamFilter(sigma_base, s2, q12, q21,
                            pi_init=q12/(q12+q21), dt=dt_id)
        pi   = filt.update_batch(S)
        pred = (pi >= 0.5).astype(int)
        acc  = max((pred==true_reg).mean(), 1-(pred==true_reg).mean())
        tp2  = ((pred==1)&(true_reg==1)).sum()
        fp2  = ((pred==1)&(true_reg==0)).sum()
        fn2  = ((pred==0)&(true_reg==1)).sum()
        pr   = tp2/(tp2+fp2) if (tp2+fp2)>0 else 0
        rc   = tp2/(tp2+fn2) if (tp2+fn2)>0 else 0
        path_accs.append(acc)
        path_prec.append(pr); path_rec.append(rc)
    accs_mean.append(np.mean(path_accs))
    accs_std.append(np.std(path_accs))
    prec_mean.append(np.mean(path_prec))
    rec_mean.append(np.mean(path_rec))
    print(f"  σ₂/σ₁={ratio:.1f}  acc={np.mean(path_accs):.1%}±{np.std(path_accs):.1%}")

accs_mean=np.array(accs_mean); accs_std=np.array(accs_std)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle('Figure 9 — Wonham Filter: Identifiability Study  '
             f'({n_paths} paths × {T_id} steps per point)',
             color=TEXT, fontsize=12)

# Left: accuracy ± std
ax = axes[0]; ax.set_facecolor(PANEL)
ax.plot(ratios, accs_mean*100, 'o-', color=TEAL, lw=2, ms=6, label='Accuracy')
ax.fill_between(ratios,
                (accs_mean-accs_std)*100,
                (accs_mean+accs_std)*100,
                color=TEAL, alpha=0.2, label='±1 std dev')
ax.axhline(95, color=ORANGE, lw=1, ls='--', alpha=0.7, label='95% threshold')
ax.axhline(80, color=RED,    lw=1, ls=':', alpha=0.5, label='80% threshold')
style_ax(ax, title='Classification Accuracy vs Volatility Separation',
         xlabel='σ₂/σ₁ ratio', ylabel='Accuracy (%)')
ax.set_ylim(50, 102); ax.legend(fontsize=9, facecolor='#222', labelcolor=TEXT, framealpha=0.8)

# Annotate paper setting
paper_idx = ratios.index(4.0)
ax.annotate(f'Paper setting\n(σ₂/σ₁=4×)\n{accs_mean[paper_idx]:.1%}',
            (4.0, accs_mean[paper_idx]*100),
            xytext=(30, -30), textcoords='offset points',
            color=ORANGE, fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))

# Right: precision and recall
ax2 = axes[1]; ax2.set_facecolor(PANEL)
ax2.plot(ratios, np.array(prec_mean)*100, 's-', color=BLUE,   lw=2, ms=6, label='Precision')
ax2.plot(ratios, np.array(rec_mean)*100,  '^-', color=PURPLE, lw=2, ms=6, label='Recall')
ax2.plot(ratios, accs_mean*100,           'o-', color=TEAL,   lw=2, ms=6, alpha=0.6, label='Accuracy')
style_ax(ax2, title='Precision & Recall vs Volatility Separation',
         xlabel='σ₂/σ₁ ratio', ylabel='Score (%)')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9, facecolor='#222', labelcolor=TEXT, framealpha=0.8)

# Add table inset
table_data = [(r, f'{a:.1%}') for r,a in zip(ratios, accs_mean)]
table_str = '\n'.join([f'σ₂/σ₁={r:.1f}×  {a}' for r,a in table_data])
ax2.text(0.98, 0.02, table_str, transform=ax2.transAxes,
         ha='right', va='bottom', color=SUBTLE, fontsize=7.5,
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#111', alpha=0.8))

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('results/plots/fig9_identifiability.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close(); print("  fig9 saved")

print("\nAll polish figures saved to results/plots/")
print("Files: fig6_confusion_matrix.png, fig7_runtime_scaling.png,")
print("       fig8_pipeline_diagram.png, fig9_identifiability.png")

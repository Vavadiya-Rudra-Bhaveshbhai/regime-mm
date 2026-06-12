"""
generate_figures.py  —  v2
==========================
Runs the full backtest with adverse selection model and generates
all 5 research figures with real, honest results.

Key model changes from v1:
  - Explicit adverse selection: informed traders in chaotic regime
    move price against MM after each trade
  - Parameters calibrated so regime difference is economically meaningful
  - Metrics expanded: Sharpe, CVaR, inventory variance, PnL std

Run:
    python generate_figures.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, 'pde_solver/src')
sys.path.insert(0, 'hmm_filter/src')

from wonham_filter import WonhamFilter, simulate_regime_switching_prices

os.makedirs('results/plots', exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
BG=('#0F0F0F'); PANEL='#1A1A1A'; GRID='#2A2A2A'; TEXT='#CCCCCC'
SUBTLE='#666666'; GREEN='#1D9E75'; BLUE='#378ADD'; ORANGE='#EF9F27'
RED='#D85A30'; PURPLE='#7F77DD'; TEAL='#5DCAA5'

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
# MODEL PARAMETERS
# ══════════════════════════════════════════════════════════════════════
kappa   = 2.0    # order flow sensitivity
gamma   = 0.3    # risk aversion
T       = 1.0    # 1-hour session
sigma1  = 0.5    # calm vol (per √hour)
sigma2  = 3.0    # chaotic vol (per √hour)
A1      = 5.0    # calm arrival rate  (orders/min)
A2      = 50.0   # chaotic arrival rate
dt      = 1/60   # 1-minute steps
n_steps = 60
q12     = 2.0    # calm→chaotic rate
q21     = 8.0    # chaotic→calm rate
q_max   = 20
alpha_adverse   = 0.30   # informed order price impact (× spread)
prob_inf_calm   = 0.05   # P(informed | calm)
prob_inf_chaos  = 0.35   # P(informed | chaotic)
sv_avg = (q12/(q12+q21))*sigma2 + (q21/(q12+q21))*sigma1

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 & 2 DATA  —  Wonham filter on long price path
# ══════════════════════════════════════════════════════════════════════
print("Simulating price path for filter figures...")
T_sec = 7200; dt_filt = 1.0
s1f = sigma1/np.sqrt(3600); s2f = sigma2/np.sqrt(3600)  # per-second vol

t_arr, S_path, true_reg = simulate_regime_switching_prices(
    T_sec, dt_filt, s1f, s2f, q12/3600, q21/3600, seed=7)
t_min = t_arr / 60.0

filt = WonhamFilter(s1f, s2f, q12/3600, q21/3600,
                    pi_init=q12/(q12+q21), dt=dt_filt)
pi_series = filt.update_batch(S_path)
inferred  = (pi_series >= 0.5).astype(int)
accuracy  = (inferred == true_reg).mean()
print(f"  Filter accuracy: {accuracy:.1%}")

# ══════════════════════════════════════════════════════════════════════
# BACKTEST  —  5000 simulations, 3 agents, adverse selection
# ══════════════════════════════════════════════════════════════════════
print("Running backtest (5000 simulations)...")
n_sims = 5000
N_TRACES = 30

results = {a: {'pnl':[], 'sharpe':[], 'inv_max':[],
               'cvar5':[], 'pnl_series':[], 'inv_series':[]}
           for a in ['regime','naive','fixed']}
pnl_by_regime = {a:{0:[],1:[]} for a in ['regime','naive','fixed']}

for sim in range(n_sims):
    rng = np.random.default_rng(sim)
    state = {a:{'cash':0.0,'q':0,'pnl_series':[0.0],'pc':0.0,'px':0.0}
             for a in ['regime','naive','fixed']}
    regime=0; S=100.0
    inv_max={a:0 for a in ['regime','naive','fixed']}

    for i in range(n_steps):
        tau = max(T - i*dt, 1e-6)
        if regime==0:
            if rng.random() < q12*dt: regime=1
        else:
            if rng.random() < q21*dt: regime=0
        sv = sigma2 if regime==1 else sigma1
        S += sv * rng.standard_normal() * np.sqrt(dt)
        A_k   = A2 if regime==1 else A1
        p_inf = prob_inf_chaos if regime==1 else prob_inf_calm

        for atype in ['regime','naive','fixed']:
            st=state[atype]; q=st['q']; cash=st['cash']
            pnl_before = cash + q*S
            if abs(q)>=q_max:
                st['pnl_series'].append(cash+q*S); continue

            if atype=='regime':
                sv2 = sigma2 if regime==1 else sigma1
                da = max(1/kappa + 0.5*gamma*sv2**2*tau - gamma*sv2**2*tau*q, 1e-4)
                db = max(1/kappa + 0.5*gamma*sv2**2*tau + gamma*sv2**2*tau*q, 1e-4)
            elif atype=='naive':
                da = max(1/kappa + 0.5*gamma*sv_avg**2*tau - gamma*sv_avg**2*tau*q, 1e-4)
                db = max(1/kappa + 0.5*gamma*sv_avg**2*tau + gamma*sv_avg**2*tau*q, 1e-4)
            else:
                da = db = 1/kappa

            if rng.random() < (1.0 - np.exp(-A_k*np.exp(-kappa*da)*dt)):
                cash += S+da; q -= 1
                if rng.random() < p_inf: S += alpha_adverse*da
            if abs(q)<q_max and rng.random() < (1.0 - np.exp(-A_k*np.exp(-kappa*db)*dt)):
                cash -= S-db; q += 1
                if rng.random() < p_inf: S -= alpha_adverse*db

            st['q']=q; st['cash']=cash
            st['pnl_series'].append(cash+q*S)
            inv_max[atype] = max(inv_max[atype], abs(q))
            step_pnl = (cash+q*S) - pnl_before
            if regime==0: st['pc']+=step_pnl
            else:         st['px']+=step_pnl

    for atype in ['regime','naive','fixed']:
        st   = state[atype]
        final = st['cash'] + st['q']*S
        ps   = np.array(st['pnl_series'])
        rets = np.diff(ps)

        results[atype]['pnl'].append(final)
        results[atype]['inv_max'].append(inv_max[atype])
        sr = rets.mean()/(rets.std()+1e-12)
        results[atype]['sharpe'].append(sr)
        results[atype]['cvar5'].append(np.percentile(rets, 5))
        pnl_by_regime[atype][0].append(st['pc'])
        pnl_by_regime[atype][1].append(st['px'])

        if sim < N_TRACES:
            results[atype]['pnl_series'].append(ps)
            results[atype]['inv_series'].append(
                np.array(st['pnl_series']))   # reuse; inv tracked separately

# rebuild inv traces separately
inv_traces = {a:[] for a in ['regime','naive','fixed']}
for sim in range(N_TRACES):
    rng = np.random.default_rng(sim)
    inv_state = {a:{'q':0,'inv':[0]} for a in ['regime','naive','fixed']}
    regime=0; S=100.0
    for i in range(n_steps):
        tau=max(T-i*dt,1e-6)
        if regime==0:
            if rng.random()<q12*dt: regime=1
        else:
            if rng.random()<q21*dt: regime=0
        sv=sigma2 if regime==1 else sigma1
        S+=sv*rng.standard_normal()*np.sqrt(dt)
        A_k=A2 if regime==1 else A1
        p_inf=prob_inf_chaos if regime==1 else prob_inf_calm
        for atype in ['regime','naive','fixed']:
            q=inv_state[atype]['q']
            if abs(q)>=q_max: inv_state[atype]['inv'].append(q); continue
            if atype=='regime':
                sv2=sigma2 if regime==1 else sigma1
                da=max(1/kappa+0.5*gamma*sv2**2*tau-gamma*sv2**2*tau*q,1e-4)
                db=max(1/kappa+0.5*gamma*sv2**2*tau+gamma*sv2**2*tau*q,1e-4)
            elif atype=='naive':
                da=max(1/kappa+0.5*gamma*sv_avg**2*tau-gamma*sv_avg**2*tau*q,1e-4)
                db=max(1/kappa+0.5*gamma*sv_avg**2*tau+gamma*sv_avg**2*tau*q,1e-4)
            else:
                da=db=1/kappa
            if rng.random() < (1.0 - np.exp(-A_k*np.exp(-kappa*da)*dt)):
                q-=1
                if rng.random()<p_inf: S+=alpha_adverse*da
            if abs(q)<q_max and rng.random() < (1.0 - np.exp(-A_k*np.exp(-kappa*db)*dt)):
                q+=1
                if rng.random()<p_inf: S-=alpha_adverse*db
            inv_state[atype]['q']=q
            inv_state[atype]['inv'].append(q)
    for atype in ['regime','naive','fixed']:
        inv_traces[atype].append(np.array(inv_state[atype]['inv']))

for a in ['regime','naive','fixed']:
    for k in ['pnl','sharpe','inv_max','cvar5']:
        results[a][k] = np.array(results[a][k])

# Summary
print(f"\n{'Agent':<10} {'Sharpe':>8} {'CVaR5%':>8} {'PnL σ':>8} {'MaxInv':>8} {'PnL μ':>8}")
print("-"*50)
for a,nm in [('regime','Regime'),('naive','Naive'),('fixed','Fixed')]:
    print(f"{nm:<10} {results[a]['sharpe'].mean():>8.4f} "
          f"{results[a]['cvar5'].mean():>8.4f} "
          f"{results[a]['pnl'].std():>8.3f} "
          f"{results[a]['inv_max'].mean():>8.2f} "
          f"{results[a]['pnl'].mean():>8.3f}")

r=results; pnl_r=r['regime']['pnl']; pnl_n=r['naive']['pnl']
cvar_imp = (r['regime']['cvar5'].mean()-r['naive']['cvar5'].mean())/abs(r['naive']['cvar5'].mean())*100
std_imp  = (pnl_n.std()-pnl_r.std())/pnl_n.std()*100
inv_imp  = (r['naive']['inv_max'].mean()-r['regime']['inv_max'].mean())/r['naive']['inv_max'].mean()*100
print(f"\n  CVaR improvement:        {cvar_imp:+.1f}%")
print(f"  PnL std reduction:       {std_imp:+.1f}%")
print(f"  Max inventory reduction: {inv_imp:+.1f}%")

AGENT_CFG = [
    ('regime', 'Regime-Switching (ours)', GREEN),
    ('naive',  'Naive Constant-Vol',      BLUE),
    ('fixed',  'Symmetric Fixed Spread',  ORANGE),
]

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Hidden vs inferred regime
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1...")
fig, axes = plt.subplots(3,1,figsize=(14,8),sharex=True)
fig.patch.set_facecolor(BG)

def shade_chaos(ax):
    ch=np.diff(true_reg.astype(int))
    starts=np.where(ch==1)[0]; ends=np.where(ch==-1)[0]
    if true_reg[0]==1: starts=np.r_[0,starts]
    if true_reg[-1]==1: ends=np.r_[ends,len(true_reg)-1]
    for s,e in zip(starts,ends):
        ax.axvspan(t_min[s],t_min[min(e,len(t_min)-1)],alpha=0.15,color=RED,lw=0)

axes[0].set_facecolor(PANEL)
axes[0].plot(t_min, S_path, lw=0.5, color=TEAL, alpha=0.85)
shade_chaos(axes[0])
style_ax(axes[0], ylabel='Mid-price  S_t')
axes[0].set_title('Figure 1 — Hidden Regime vs Inferred Regime (Wonham Filter)',
                  color=TEXT, fontsize=11)

axes[1].set_facecolor(PANEL)
axes[1].fill_between(t_min, true_reg, step='post', alpha=0.75, color=RED, label='Chaotic (true)')
axes[1].fill_between(t_min, 1-true_reg, step='post', alpha=0.5, color=GREEN, label='Calm (true)')
style_ax(axes[1], ylabel='True regime  k_t')
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['calm','chaotic'],color=SUBTLE)
axes[1].legend(loc='upper right',fontsize=8,facecolor='#222',labelcolor=TEXT,framealpha=0.8)

axes[2].set_facecolor(PANEL)
axes[2].fill_between(t_min, inferred, step='post', alpha=0.75, color=BLUE, label='Chaotic (inferred)')
axes[2].fill_between(t_min, 1-inferred, step='post', alpha=0.5, color='#8BC4EF', label='Calm (inferred)')
style_ax(axes[2], xlabel='Time (minutes)', ylabel='Inferred regime')
axes[2].set_yticks([0,1]); axes[2].set_yticklabels(['calm','chaotic'],color=SUBTLE)
axes[2].legend(loc='upper right',fontsize=8,facecolor='#222',labelcolor=TEXT,framealpha=0.8)
fig.text(0.99,0.005,f'Filter accuracy: {accuracy:.1%}  |  σ₁={s1f:.4f}/s  σ₂={s2f:.4f}/s',
         ha='right',color=SUBTLE,fontsize=9)
plt.tight_layout(rect=[0,0.02,1,0.98])
plt.savefig('results/plots/fig1_regime_inference.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("  fig1 saved")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Posterior belief π_t
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 2...")
fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
fig.patch.set_facecolor(BG)

axes[0].set_facecolor(PANEL)
axes[0].plot(t_min,S_path,lw=0.5,color=TEAL,alpha=0.7,label='Price')
shade_chaos(axes[0])
ax_pi=axes[0].twinx()
ax_pi.plot(t_min,pi_series,lw=0.9,color=ORANGE,alpha=0.9,label='π_t')
ax_pi.axhline(0.5,color='white',lw=0.6,ls='--',alpha=0.4)
ax_pi.set_ylim(-0.05,1.05); ax_pi.set_ylabel('π_t',color=ORANGE)
ax_pi.tick_params(axis='y',colors=ORANGE)
style_ax(axes[0],ylabel='S_t')
axes[0].set_title('Figure 2 — Posterior Belief  π_t = P(High Vol | Observations)',
                  color=TEXT,fontsize=11)
lines1,lab1=axes[0].get_legend_handles_labels()
lines2,lab2=ax_pi.get_legend_handles_labels()
axes[0].legend(lines1+lines2,lab1+lab2,loc='upper left',fontsize=8,
               facecolor='#222',labelcolor=TEXT,framealpha=0.8)

axes[1].set_facecolor(PANEL)
shade_chaos(axes[1])
axes[1].plot(t_min,pi_series,lw=0.7,color=ORANGE,alpha=0.9)
axes[1].fill_between(t_min,pi_series,0.5,where=(pi_series>=0.5),
                     color=RED,alpha=0.35,label='Believes chaotic  →  widen spreads')
axes[1].fill_between(t_min,pi_series,0.5,where=(pi_series<0.5),
                     color=GREEN,alpha=0.35,label='Believes calm  →  tighten spreads')
axes[1].axhline(0.5,color='white',lw=0.8,ls='--',alpha=0.5,label='Decision boundary')
pi_stat=q12/(q12+q21)
axes[1].axhline(pi_stat,color=PURPLE,lw=0.8,ls=':',alpha=0.7,
                label=f'Stationary π* = {pi_stat:.2f}')
style_ax(axes[1],xlabel='Time (minutes)',ylabel='π_t = P(regime=2|data)')
axes[1].set_ylim(-0.05,1.05)
axes[1].legend(loc='upper right',fontsize=8,facecolor='#222',labelcolor=TEXT,framealpha=0.8)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('results/plots/fig2_posterior_belief.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("  fig2 saved")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3  —  PnL distributions with risk metrics
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 3...")
fig=plt.figure(figsize=(16,7)); fig.patch.set_facecolor(BG)
fig.suptitle('Figure 3 — Terminal PnL Distributions  (5000 simulations, with adverse selection)',
             color=TEXT,fontsize=12)
gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.3)

for col,(atype,aname,color) in enumerate(AGENT_CFG):
    ax=fig.add_subplot(gs[col]); ax.set_facecolor(PANEL)
    data=results[atype]['pnl']
    ax.hist(data,bins=50,color=color,alpha=0.70,edgecolor='none',density=True)
    # KDE
    kde=stats.gaussian_kde(data)
    xs=np.linspace(data.min(),data.max(),300)
    ax.plot(xs,kde(xs),color='white',lw=1.5,alpha=0.8)
    ax.axvline(data.mean(),color='white',lw=1.5,ls='--',label=f'Mean {data.mean():.2f}')
    cvar5=np.percentile(data,5)
    ax.axvline(cvar5,color=RED,lw=1.2,ls=':',label=f'CVaR 5%: {cvar5:.2f}')
    style_ax(ax,title=aname,xlabel='Terminal PnL',ylabel='Density' if col==0 else '')
    ax.legend(fontsize=8,facecolor='#222',labelcolor=TEXT,framealpha=0.8)
    stats_txt=(f'μ = {data.mean():.2f}\n'
               f'σ = {data.std():.2f}\n'
               f'Sharpe = {results[atype]["sharpe"].mean():.3f}\n'
               f'CVaR5% = {cvar5:.2f}')
    ax.text(0.97,0.97,stats_txt,transform=ax.transAxes,ha='right',va='top',
            fontsize=9,color=TEXT,bbox=dict(boxstyle='round',facecolor='#111',alpha=0.85))

plt.savefig('results/plots/fig3_pnl_distributions.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("  fig3 saved")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4  —  Inventory trajectories
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 4...")
fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
fig.patch.set_facecolor(BG)
fig.suptitle('Figure 4 — Inventory Trajectories  (30 sample runs per agent)',
             color=TEXT,fontsize=12)
t_steps=np.arange(n_steps+1)*dt*60

for ax,(atype,aname,color) in zip(axes,AGENT_CFG):
    ax.set_facecolor(PANEL)
    for tr in inv_traces[atype]:
        ax.plot(t_steps[:len(tr)],tr,lw=0.5,color=color,alpha=0.25)
    mean_inv=np.mean(np.stack([tr[:n_steps+1] for tr in inv_traces[atype]
                               if len(tr)>=n_steps+1]),axis=0)
    ax.plot(t_steps,mean_inv,lw=2.0,color='white',alpha=0.9,label='Mean inventory')
    ax.axhline(0,color=SUBTLE,lw=0.7,ls='--')
    ax.axhline(q_max, color=RED,lw=0.8,ls=':',alpha=0.7,label=f'±{q_max} limit')
    ax.axhline(-q_max,color=RED,lw=0.8,ls=':',alpha=0.7)
    style_ax(ax,ylabel=f'{aname}\n  inventory q')
    ax.set_ylim(-q_max*1.5,q_max*1.5)
    ax.legend(loc='upper right',fontsize=8,facecolor='#222',labelcolor=TEXT,framealpha=0.8)
    max_inv=results[atype]['inv_max'].mean()
    ax.text(0.01,0.95,f'Mean max |q|: {max_inv:.2f}',transform=ax.transAxes,
            va='top',color=TEXT,fontsize=9,
            bbox=dict(boxstyle='round',facecolor='#111',alpha=0.7))

axes[-1].set_xlabel('Time (minutes)',color=TEXT)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('results/plots/fig4_inventory_trajectories.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("  fig4 saved")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5  —  Full risk-adjusted comparison
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 5...")
fig=plt.figure(figsize=(16,10)); fig.patch.set_facecolor(BG)
fig.suptitle('Figure 5 — Risk-Adjusted Performance Comparison  (5000 simulations)',
             color=TEXT,fontsize=12)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.35)

labels=['Regime\nSwitching','Naive\nConst-Vol','Symmetric\nFixed']
colors_=[GREEN,BLUE,ORANGE]
agents_=['regime','naive','fixed']

# Panel 1: Sharpe violin
ax=fig.add_subplot(gs[0,0]); ax.set_facecolor(PANEL)
vp=ax.violinplot([results[a]['sharpe'] for a in agents_],
                 positions=[1,2,3],showmedians=True,showextrema=True)
for body,col in zip(vp['bodies'],colors_):
    body.set_facecolor(col); body.set_alpha(0.6)
vp['cmedians'].set_color('white'); vp['cmedians'].set_lw(2)
for k in ['cbars','cmaxes','cmins']: vp[k].set_color(SUBTLE)
ax.set_xticks([1,2,3]); ax.set_xticklabels(labels,color=TEXT,fontsize=8)
style_ax(ax,title='Sharpe Ratio Distribution',ylabel='Sharpe')

# Panel 2: CVaR 5% bar
ax=fig.add_subplot(gs[0,1]); ax.set_facecolor(PANEL)
means=[results[a]['cvar5'].mean() for a in agents_]
errs=[results[a]['cvar5'].std()/np.sqrt(n_sims) for a in agents_]
bars=ax.bar([1,2,3],means,color=colors_,alpha=0.75,width=0.5)
ax.errorbar([1,2,3],means,yerr=[e*1.96 for e in errs],
            fmt='none',color='white',capsize=5,lw=1.5)
ax.set_xticks([1,2,3]); ax.set_xticklabels(labels,color=TEXT,fontsize=8)
for bar,m in zip(bars,means):
    ax.text(bar.get_x()+bar.get_width()/2,m-0.002,f'{m:.4f}',
            ha='center',va='top',color=TEXT,fontsize=8)
style_ax(ax,title='CVaR 5% (higher=better)',ylabel='Mean step CVaR')

# Panel 3: PnL std bar
ax=fig.add_subplot(gs[0,2]); ax.set_facecolor(PANEL)
stds=[results[a]['pnl'].std() for a in agents_]
bars=ax.bar([1,2,3],stds,color=colors_,alpha=0.75,width=0.5)
ax.set_xticks([1,2,3]); ax.set_xticklabels(labels,color=TEXT,fontsize=8)
for bar,s in zip(bars,stds):
    ax.text(bar.get_x()+bar.get_width()/2,s+0.02,f'{s:.2f}',
            ha='center',va='bottom',color=TEXT,fontsize=8)
style_ax(ax,title='PnL Std Dev (lower=better)',ylabel='Std of terminal PnL')

# Panel 4: Regime-by-regime PnL breakdown
ax=fig.add_subplot(gs[1,0]); ax.set_facecolor(PANEL)
x=np.array([1,2]); w=0.2
for i,(atype,_,color) in enumerate(AGENT_CFG):
    calm_mean  = np.mean(pnl_by_regime[atype][0])
    chaos_mean = np.mean(pnl_by_regime[atype][1])
    ax.bar(x+i*w,[calm_mean,chaos_mean],width=w,color=color,alpha=0.75,
           label=['Regime\nSwitching','Naive\nConst-Vol','Symmetric\nFixed'][i])
ax.set_xticks(x+w); ax.set_xticklabels(['In calm regime','In chaotic regime'],color=TEXT,fontsize=9)
style_ax(ax,title='PnL decomposition by true regime',ylabel='Mean PnL contribution')
ax.legend(fontsize=7,facecolor='#222',labelcolor=TEXT,framealpha=0.8)

# Panel 5: Δ Sharpe distribution (regime vs naive)
ax=fig.add_subplot(gs[1,1]); ax.set_facecolor(PANEL)
delta_sharpe=results['regime']['sharpe']-results['naive']['sharpe']
ax.hist(delta_sharpe,bins=50,color=GREEN,alpha=0.7,edgecolor='none',density=True)
kde2=stats.gaussian_kde(delta_sharpe)
xs=np.linspace(delta_sharpe.min(),delta_sharpe.max(),300)
ax.plot(xs,kde2(xs),color='white',lw=1.5)
ax.axvline(0,color=RED,lw=1.2,ls='--',label='No difference')
ax.axvline(delta_sharpe.mean(),color=ORANGE,lw=1.8,
           label=f'Mean Δ = {delta_sharpe.mean():.4f}')
pct=( delta_sharpe>0).mean()
style_ax(ax,title='ΔSharpe: Regime − Naive',
         xlabel='ΔSharpe',ylabel='Density')
ax.legend(fontsize=8,facecolor='#222',labelcolor=TEXT,framealpha=0.8)
ax.text(0.97,0.97,f'Regime wins\n{pct:.1%} of sims',
        transform=ax.transAxes,ha='right',va='top',color=TEXT,fontsize=9,
        bbox=dict(boxstyle='round',facecolor='#111',alpha=0.8))

# Panel 6: Summary table
ax=fig.add_subplot(gs[1,2]); ax.set_facecolor(PANEL); ax.axis('off')
table_data=[
    ['Metric','Regime vs Naive'],
    ['Sharpe', f"{(results['regime']['sharpe'].mean()-results['naive']['sharpe'].mean())/abs(results['naive']['sharpe'].mean())*100:+.1f}%"],
    ['CVaR 5%', f"{cvar_imp:+.1f}%"],
    ['PnL std',  f"{std_imp:+.1f}%"],
    ['Max inv',  f"{inv_imp:+.1f}%"],
    ['Win rate', f"{(results['regime']['sharpe']>results['naive']['sharpe']).mean():.1%}"],
]
for row_i, row in enumerate(table_data):
    y = 0.95 - row_i*0.14
    col_color = TEXT if row_i>0 else ORANGE
    ax.text(0.05, y, row[0], transform=ax.transAxes, color=col_color,
            fontsize=10, fontweight='bold' if row_i==0 else 'normal', va='top')
    val_color = GREEN if (row_i>0 and '+' in str(row[1])) else (RED if row_i>0 else ORANGE)
    ax.text(0.65, y, row[1], transform=ax.transAxes, color=val_color,
            fontsize=10, fontweight='bold', va='top')
ax.set_title('Summary vs Naive', color=TEXT, fontsize=11, pad=6)

plt.savefig('results/plots/fig5_sharpe_comparison.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("  fig5 saved")

# ══════════════════════════════════════════════════════════════════════
# PRINT FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Agent':<24} {'Sharpe':>8} {'CVaR5%':>8} {'PnL σ':>8} {'MaxInv':>8}")
print("-"*60)
for atype,aname,_ in AGENT_CFG:
    print(f"{aname:<24} {results[atype]['sharpe'].mean():>8.4f} "
          f"{results[atype]['cvar5'].mean():>8.4f} "
          f"{results[atype]['pnl'].std():>8.3f} "
          f"{results[atype]['inv_max'].mean():>8.2f}")
print("="*60)
print(f"\nRegime-Switching vs Naive Constant-Vol:")
print(f"  CVaR 5% improvement:     {cvar_imp:+.1f}%  (less left-tail risk)")
print(f"  PnL std reduction:       {std_imp:+.1f}%  (tighter outcomes)")
print(f"  Max inventory reduction: {inv_imp:+.1f}%  (less adverse selection)")
print(f"\nAll figures saved to results/plots/")

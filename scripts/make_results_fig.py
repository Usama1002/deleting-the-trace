"""Headline results figure from the real logged numbers (E1 + E2). Vector PDF."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

e1=json.load(open("results/e1_cot_suppression.json"))["conditions"]
e2=json.load(open("results/e2b_monitor_gemma.json"))

NAVY="#1B2A4A"; TEAL="#2A9D8F"; CORAL="#E76F51"; GRAY="#9AA5B1"
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.sprsine.top":False} if False else {"font.size":11})

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.2,3.4))

# Panel (a): reasoning-channel tokens, plain vs forged (unsafe), plus that the action still fires.
cot_plain=e1["no_forge"]["unsafe"]["cot_tokens_mean"]
cot_forge=e1["forge"]["unsafe"]["cot_tokens_mean"]
bars=ax1.bar(["plain","forged"],[cot_plain,cot_forge],color=[TEAL,GRAY],width=0.55,edgecolor=NAVY,linewidth=1.2)
ax1.set_ylabel("reasoning-channel tokens (mean)")
ax1.set_ylim(0,60)
for b,v in zip(bars,[cot_plain,cot_forge]):
    ax1.text(b.get_x()+b.get_width()/2,v+1.2,f"{v:.1f}",ha="center",va="bottom",fontweight="bold")
ax1.text(1.0,26,"unsafe action\nfires 100%\nin both\nconditions",ha="center",va="center",fontsize=10,
         style="italic",color=NAVY)
ax1.set_title("(a) control-token injection deletes the CoT",fontsize=11,color=NAVY)
for s in ("top","right"): ax1.spines[s].set_visible(False)

# Panel (b): monitor detection on malicious requests, plain vs forged.
mons=["rule\n(CoT)","LLM\n(CoT)","action"]
plain=[e2["unsafe_noforge"]["rule_cot"],e2["unsafe_noforge"]["llm_cot_gemma"],e2["unsafe_noforge"]["action_mon"]]
forge=[e2["unsafe_forge"]["rule_cot"],e2["unsafe_forge"]["llm_cot_gemma"],e2["unsafe_forge"]["action_mon"]]
import numpy as np
x=np.arange(len(mons)); w=0.38
b1=ax2.bar(x-w/2,plain,w,label="plain",color=TEAL,edgecolor=NAVY,linewidth=1.1)
b2=ax2.bar(x+w/2,forge,w,label="forged",color=CORAL,edgecolor=NAVY,linewidth=1.1)
ax2.set_xticks(x); ax2.set_xticklabels(mons)
ax2.set_ylabel("detection rate on malicious requests")
ax2.set_ylim(0,1.08)
for bars in (b1,b2):
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{b.get_height():.2f}",ha="center",va="bottom",fontsize=9)
ax2.legend(frameon=False,loc="upper right",ncol=1,fontsize=9.5)
ax2.set_title("(b) forge blinds CoT monitors, not the action monitor",fontsize=11,color=NAVY)
for s in ("top","right"): ax2.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig("paper/figures/results.pdf",bbox_inches="tight")
fig.savefig("paper/figures/results.png",dpi=150,bbox_inches="tight")
print("saved paper/figures/results.pdf")
print("a:",cot_plain,cot_forge,"| b plain:",plain,"forge:",forge)

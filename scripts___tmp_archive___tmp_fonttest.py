import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa

plt.style.use(["science", "no-latex"])

variants = {
    "A_list_family": {"font.family": ["Times New Roman", "SimSun"]},
    "B_simSun_first": {"font.family": "serif", "font.serif": ["SimSun", "Times New Roman"]},
}
for name, rc in variants.items():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    plt.rcParams.update(rc)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_title("测试中文标题 EF 百分点")
    ax.set_xlabel("采样帧数")
    fig.tight_layout()
    fig.savefig(rf"F:\EchoCLIP_paper_build\_fonttest_{name}.png", dpi=150)
    plt.close(fig)
    print("done", name)

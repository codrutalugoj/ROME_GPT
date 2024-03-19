# %%
import matplotlib.pyplot as plt



# %%
def plot_fig1(indirect_effect, prompt):
    
    ax = plt.imshow(indirect_effect, cmap="RdBu")
    plt.colorbar(ax)
    plt.yticks(range(len(prompt)), prompt)
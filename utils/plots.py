import matplotlib.pyplot as plt


def plot_tmp(TMP, y, a, out, name):
        # Detach tensors from computation graph and move to CPU for plotting
    y_detached = y.detach().cpu().numpy()
    TMP_detached = TMP.detach().cpu().numpy()
    a_detached = a.detach().cpu().numpy()
    out_detached = out.detach().cpu().numpy()
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 3))
    
    ax1.plot(y_detached[0, :, 0], label='a gt' + str(a_detached[0]))
    ax1.plot(TMP_detached[0, :, 0], label='a' + str(out_detached[0]))
    ax1.legend()
    
    ax2.plot(y_detached[2, :, 0], label='a gt' + str(a_detached[2]))
    ax2.plot(TMP_detached[2, :, 0], label='a' + str(out_detached[2]))
    ax2.legend()
    
    ax3.plot(y_detached[3, :, 0], label='a gt' + str(a_detached[3]))
    ax3.plot(TMP_detached[3, :, 0], label='a' + str(out_detached[3]))
    ax3.legend()
    
    # Save plot to file
    plt.savefig(name)
    plt.close()

    # Ensure garbage collection of the figure
    fig.clf()
    plt.close(fig)
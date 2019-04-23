
def plot_amplitudes(amplitudes, energies, ax=None):
    """
    Description
    -----------
    Makes a nice plot of the probabilities for each state and its energy
    
    Parameters
    ----------
    :param amplitudes: (array/list) the probabilites to find the state
    :param energies:   (array/list) The energy of that state
    :ax:               (matplotlib axes object) The canvas to draw on
    """
    if ax == None:
        fig, ax = plt.subplots()
    # normalizing energies
    energies = np.array(energies)
    energies /= max(abs(energies))
    
    format_strings = ('{0:00b}', '{0:01b}', '{0:02b}', '{0:03b}', '{0:04b}', '{0:05b}')
    nqubits = int(np.log2(len(energies)))
    
    # create labels
    labels = [r'$\left|' + format_strings[nqubits].format(i) + r'\right>$' for i in range(len(amplitudes))]
    y_pos = np.arange(len(amplitudes))
    width = 0.35
    ax.bar(y_pos, amplitudes**2, width, label=r'$|Amplitude|^2$')
    
    ax.bar(y_pos+width, -energies, width, label="-Energy")
    ax.set_xticks(y_pos+width/2, minor=False)
    ax.set_xticklabels(labels, minor=False)
#    plt.ylabel("Amplitude")
    ax.set_xlabel("State")
    ax.grid(linestyle='--')
    ax.legend()
#    plt.show()

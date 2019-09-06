.. _faq:

Frequently asked Questions
==========================


Why is mixing hamiltonian :math:`-\sum_i \hat{X}_i` instead of :math:`\sum_i \hat{X}_i` ?
    In the original QAOA paper Farhi et al use :math:`-\sum_i \hat{X}_i` as
    the mixer hamiltonian and :math:`\left|+ \cdots +\right>` as the initial
    state. This is the `maximum` energy eigenstate of this mixer hamiltonian.
    Hence the original QAOA algorithm is good for finding the `maximum` energy
    eigenstate of the cost hamiltonian. However in the context of optimisation
    people usually talk about `minimization`. So we decided to change the
    mixer hamiltonian to :math:`-\sum_i \hat{X}_i` and use
    :math:`\left|+ \cdots +\right>` as the initial state. This implies, that
    our implementation is good for finding `minimum` energy eigenstates.


Where does that magic time of ``0.7 * n_steps`` come from?
    The ``.linear_ramp_from_hamiltonian()`` parameters are inspired by the idea
    of QAOA being discretized adiabatic annealing with a linear ramp schedule,
    i.e. :math:`s(t) = \frac{t}{\tau}`, where :math:`\tau` is the total
    annealing time. For such a discretized schedule we need to specify two
    numbers: The total annealing time :math:`\tau` and the step width
    :math:`\Delta t` or equivalently the total annealing time :math:`\tau` and
    the number of steps :math:`n_{\textrm{steps}}` (also called `p` in the
    context of QAOA). A good discretized annealing schedule has to strike a
    balance between a long annealing time :math:`\tau` and a small step width
    :math:`\Delta t = \frac{\tau}{n_{\textrm{steps}}}`. We found in numerical
    experiments that :math:`\Delta t = 0.7 = \frac{\tau}{n_{\textrm{steps}}}`
    does strike such a good balance, at least for system sizes up to 10
    qubits. For larger systems or smaller energy gaps smaller it might be
    neccesary to choose smaller values of :math:`\Delta t`


What Sine and Cosine transforms do you use for the Fourier Parameters?
    We use

    .. math::

        \gamma_i = 2 \sum_{k=0}^{q-1} u_k
                    \sin \left[
                            (2k - 1) k \frac{\pi}{2p}
                         \right]

        \beta_i = 2 \sum_{k=0}^{q-1} v_k
                    \cos \left[
                            (2k - 1) k \frac{\pi}{2p}
                         \right].

    as compared to the original paper by Leo Zhou et al, where they use

    .. math::

        \gamma_i = \sum_{k=0}^q u_k
                    \sin \left[
                            \left(k - \frac{1}{2}\right)
                            \left(i-\frac{1}{2}\right)
                            \frac{\pi}{p}
                         \right]

        \beta_i = \sum_{k=0}^q v_k
                    \cos \left[
                            \left(k - \frac{1}{2}\right)
                            \left(i-\frac{1}{2}\right)
                            \frac{\pi}{p}
                         \right]

    for their DCT/DST.

    This is simply, because the first is the default choice of
    ``scipy.fftpack.dct`` or ``scipy.fftpack.dst`` and doesn't make a big
    difference in terms of performance or expressivity.


Where does the factor of 2 in the rotation angles in ``qaoa.cost_function._qaoa_cost_ham_rotation`` and ``qaoa.cost_function._qaoa_mixing_ham_rotation`` come from?
    The Pauli Rotation gates are defined as :math:`R_x(\theta) = e^{- \frac{-i \theta}{2} \sigma_x}` and for :math:`R_y` and :math:`R_z` similarly. So to implement :math:`e^{-i \phi \sigma_x}` we have to do :math:`R_x(2 \phi)`



What is the difference between ``base_numshots`` and ``n_shots`` in ``PrepareAndMeasureOnQVM`` and ``QAOACostFunctionOnQVM``?
    The cost functions created by ``PrepareAndMeasureOnQVM`` and
    ``QAOACostFunctionOnQVM`` both make use of Quils
    `parametric programs <http://docs.rigetti.com/en/latest/basics.html?programs#parametric-compilation>`_. This means, the circuit is
    compiled once before the optimisation starts and then only the variable
    parameters are changed by the optimiser. Unfortunately, the number of
    repetitions to run the circuit can only be set once before compilation via
    `Program.wrap_in_numshots_loop(base_numshots) <http://docs.rigetti.com/en/latest/apidocs/autogen/pyquil.quil.Program.wrap_in_numshots_loop.html>`_. If running on the QVM this has the effect, that the Wavefunction is calculated once, but `base_numshots` samples are taken from it. On the QPU the same program has to be run `base_numshots` times.
    If you want to change the number of shots taken during the optimisation you
    can only repeat the program `n_shots` times. So `base_numshots` is a hard compiled number base number of shots that is taken after each execution of the program and `n_shots` is a multiplier of that saying how often to run the program. If you don't want to change to number of shots to take during the optimisation you should set `n_shots=1` and `base_numshots` to whatever number of shots you want to take. This is much faster, than the other way around.


Why do I have to instantiate a parameter object before creating a cost function
    Technical reasons

"""
Holds a class that manages QVM processes to get multiple of these going
"""

import pyquil.api as api
import subprocess


def get_open_port():
    """Return unused port to start QVM on
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


class qvm_process():
    """
    Description
    -----------
        Context manager class to hold a qvm process. Each process runs on a unique
        port to allow for mutliple QVMs to run at the same time

    Attributes
    ----------
        ivar sim:      A pyquil.api.WavefunctionSimulator connection
        ivar qvm:      A pyquil.api.get_qc qvm connection

    Example
    -------
        >>> with qvm_process("9q-generic-qvm", as_qvm=True) as proc:
        >>>     sim = proc.sim
        >>>     sim.wavefunction(quil_programm)
    """

    def __init__(self, *qvm_args, **qvm_kwargs):
        self._qvm_args = qvm_args
        self._qvm_kwargs = qvm_kwargs

        self._qvm_port = get_open_port()
        self._qvm_proc = subprocess.Popen(['qvm', '-S', '-p',
                                           str(self._qvm_port)],
                                          stdin=None, stdout=None,
                                          stderr=None, close_fds=True)
        print("Started QVM on port {port}".format(port=self._qvm_port))
        qvm_endpoint = "http://localhost:" + str(self._qvm_port)
        qvm_connection = api.ForestConnection(sync_endpoint=qvm_endpoint)
        self.qvm = api.get_qc(*self._qvm_args, connection=qvm_connection, **self._qvm_kwargs)
        self.sim = api.WavefunctionSimulator(connection=qvm_connection)

        self._quilc_port = get_open_port()
        self._quilc_proc = subprocess.Popen(['quilc', '-S', '-p]',
                                             str(self._quilc_port)],
                                            stdin=None, stdout=None,
                                            stderr=None, close_fds=True)
        print("Started Quil Compiler on {port}".format(port=self._quilc_port))
        quilc_endpoint = "http://localhost:" + str(self._quilc_port)
#        quilc_connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()

    def kill(self):
        """
        Kills the qvm process
        """
        print("Killed the QVM on port {port}".format(port=self._qvm_port))
        print("Killed the quilc on port {port}".format(port=self._quilc_port))
        self._qvm_proc.kill()
        self._quilc_proc.kill()

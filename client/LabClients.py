import threading

from BasicSAClient import BasicSAClient, sendRequest
from TurboClient import TurboClient
from HeteroSAClient import HeteroSAClient
from CSAClient import CSAClient

def runOneClient(mode, k):
    if mode == 0: # BasicSA
        client = BasicSAClient()
        for _ in range(k):
            client.setUp()
            client.advertiseKeys()
            client.shareKeys()
            client.maskedInputCollection()
            client.unmasking()

    elif mode == 1: # Turbo
        client = TurboClient()
        for _ in range(k):
            group = client.setUp()
            if group != 0:
                client.turbo()
            client.turbo_value()
            client.turbo_final()

    #elif mode == 2: # BREA

    elif mode == 3: # HeteroSA
        client = HeteroSAClient()
        for _ in range(k):
            client.setUp()
            client.advertiseKeys()
            client.shareKeys()
            client.maskedInputCollection()
            client.unmasking()

    elif mode == 4: # BasicCSA
        client = CSAClient(isBasic = True)
        for _ in range(3):
            client.setUp()
            client.shareRandomMasks()
            client.sendSecureWeight()

    elif mode == 5: # FullCSA
        client = CSAClient(isBasic = False)
        for _ in range(3):
            client.setUp()
            client.shareRandomMasks()
            client.sendSecureWeight()


if __name__ == "__main__":
    # args
    k = 3       # rounds
    n = 4       # number of users
    mode = 0
    """ mode
    0: BasicSA Client
    1: Turbo Client
    2: BREA Client
    3: HeteroSA Client
    4: BasicCSA Client
    5: FullCSA Client
    """

    host = 'localhost'
    port = 6000
    sendRequest(host, port, mode, {'n': n, 'k': k})

    # thread
    for _ in range(n):
        thread = threading.Thread(target=runOneClient, args=(mode, k))
        thread.start()
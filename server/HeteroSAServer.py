import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from MainServer import MainServer
from BasicSA import getCommonValues
from HeteroSAg import SS_Matrix
from dto.HeteroSetupDto import HeteroSetupDto, HeteroKeysRequestDto
from CommonValue import BasicSARound
import learning.federated_main as fl
import SecureProtocol as sp

users_keys = {}
n = 4 # expected (== G * perGroup == G * G)
threshold = 1
R = 0
B = [] # The SS Matrix B
G = 2 # group num
perGroup = 2
usersNow = 0
segment_yu = {}

# broadcast common value
def setUp():
    global n, threshold, R, B, usersNow, G, perGroup

    tag = BasicSARound.SetUp.name
    port = BasicSARound.SetUp.value
    
    server = MainServer(tag, port, n)
    server.start()

    usersNow = server.userNum
    n = usersNow
    threshold = usersNow - 2 # temp

    commonValues = getCommonValues()
    R = commonValues["R"]
    g = commonValues["g"]
    p = commonValues["p"]

    # Segment Grouping Strategy: G x G Segment selection matrix B
    B = SS_Matrix(G)
    
    response = []
    for i in range(G):
        for j in range(perGroup):
            response_ij = HeteroSetupDto(
                n, threshold, g, p, R, i, i*perGroup + j, B, G
            )._asdict()
            response.append(response_ij)
    server.foreachIndex(response)

def advertiseKeys():
    global users_keys, usersNow, G

    tag = BasicSARound.AdvertiseKeys.name
    port = BasicSARound.AdvertiseKeys.value
    server = MainServer(tag, port, usersNow)
    server.start()
    usersNow = server.userNum

    # requests example: {"group":, "index":, "c_pk":"VALUE", "s_pk": "VALUE"}
    requests = server.requests

    # make response
    # response example: {"INDEX": ("c_pk":"VALUE", "s_pk": "VALUE")}
    for request in requests:
        requestData = request[1]  # (socket, data)
        index = requestData["index"]
        users_keys[index] = {"c_pk": requestData["c_pk"], "s_pk": requestData["s_pk"]}
    server.broadcast(users_keys)


def shareKeys():
    global usersNow

    tag = BasicSARound.ShareKeys.name
    port = BasicSARound.ShareKeys.value
    server = MainServer(tag, port, usersNow)
    server.start()
    usersNow = server.userNum

    # (one) request example: {0: [(0, 0, e00), (0, 1, e01) ... ]}
    # requests example: [{0: [(0, 0, e00), ... ]}, {1: [(1, 0, e10), ... ]}, ... ]
    requests = server.requests

    # response example: { 0: [e00, e10, e20, ...], 1: [e01, e11, e21, ... ] ... }
    response = {}
    requests_euv = []
    for request in requests:
        requestData = request[1]  # (socket, data)
        for idx, data in requestData.items(): #only one
            response[idx] = {}  # make dic
            requests_euv.append(data)
    for request in requests_euv:
        for (u, v, euv) in request:
            try:
                response[str(v)][u] = euv
            except KeyError:  # drop-out
                print("KeyError: drop-out!")
                pass

    server.foreach(response)

surviving_users = []

def maskedInputCollection():
    global segment_yu, usersNow, G, surviving_users

    tag = BasicSARound.MaskedInputCollection.name
    port = BasicSARound.MaskedInputCollection.value
    server = MainServer(tag, port, usersNow)
    server.start()
    usersNow = server.userNum

    # if u3 dropped
    # (one) request example: {"group":0, "index":0, "segment_yu":{0: y0, 1: y1}}
    requests = server.requests

    # response example: { "users": [0, 1, 2 ... ] }
    segment_yu = {i: {j: [] for j in range(G)} for i in range(G)} # i: segment level, j: quantization level
    for request in requests:
        requestData = request[1]  # (socket, data)
        index = int(requestData["index"])
        surviving_users.append(index)
        for i, segment in requestData["segment_yu"].items():
            for q, yu in segment.items():
                # yu_ = fl.dic_of_list_to_weights(yu)
                segment_yu[int(i)][int(q)].append(yu)

    server.broadcast({"users": surviving_users})
    print(f'surviving_users: {surviving_users}')
    print(f'segment_yu: {segment_yu}')

if __name__ == "__main__":
    setUp()
    advertiseKeys()
    shareKeys()
    maskedInputCollection()
    
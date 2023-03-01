# Lab-SA Basic SA for Federated Learning
import random, math
import SecureProtocol as sp

# Get common values for server set-up: n, t, ...
def getCommonValues():
    # R: domain
    # g: generator, p: prime
    R = 100 #temp
    commonValues = {"g": sp.g, "p": sp.p, "R": R}
    return commonValues

# Generate two key pairs
def generateKeyPairs():
    c_pk, c_sk = sp.generateKeyPair(sp.g, sp.p)
    s_pk, s_sk = sp.generateKeyPair(sp.g, sp.p)
    return (c_pk, c_sk), (s_pk, s_sk)

def stochasticQuantization(weights, q, p):
    # weights = local model of user
    # q = quantization level
    # p = large prime

    quantized = []
    for x in weights:
        floor_qx = math.floor(q * x)
        selected = int(random.choices(
            population = [floor_qx / q, (floor_qx + 1) / q],
            weights = [1 - (q * x - floor_qx), q * x - floor_qx],
            k = 1 # select one
        )[0] * q)
        if selected < 0:
            selected = selected + p
        quantized.append(selected)

    return quantized

def convertToRealDomain(weights, q, p):
    # weights = local model of user
    # q = quantization level
    # p = large prime

    real_numbers = []
    m = (p - 1) / 2
    for x in weights:
        if 0 <= x and x < m:
            real_numbers.append(x / q)
        else: # (p-1)/2 <= x < p
            real_numbers.append((x - p) / q)
    return real_numbers

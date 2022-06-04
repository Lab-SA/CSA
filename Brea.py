import random, copy
import numpy as np
import learning.federated_main as fl
import learning.models_helper as mhelper
from BasicSA import stochasticQuantization
from Turbo import generateLagrangePolynomial

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# make N theta_i
# [호출] : 서버
# [인자] : n(사용자수), q
# [리턴] : th(각 client들에 대한 theta list)
def make_theta(n, q):
    th = []
    for i in range(n):
        th.append(random.randint(1, q))
    return th


# Secret polynomium coefficients r_ij
# [호출] : 클라이언트
# [인자] : T(threshold), q
# [리턴] : rij(사용자 i가 랜덤 생성한 계수)(j: 1~T)
def generate_rij(T, q):
    rij = [0, ] # not using first index (0)
    for j in range(T):
        rij.append(random.randint(1, q))
    return rij


# share를 생성하기 위한 다항식 f
# [인자] : theta(자신의 theta), w(weights. quantization한 값), T(threshold), rij(get_rij()의 리턴값)
# [리턴] : y
def f(theta, w, T, rij):
    y = w
    for j in range(1, T+1):
        y = y + (rij[j] * (theta ** j))
    return y


# make shares
# [호출] : 클라이언트
# [인자] : w, theta_list(서버가 알려준 theta list), T, rij_list
# [리턴] : shares (다른 사용자들에게 보낼 share list)
def make_shares(w, theta_list, T, rij_list):
    shares = []
    for theta in theta_list:
        s = []
        for k in w:
            s.append(f(theta, k, T, rij_list))
        shares.append(s)
    return shares


# make commitment
# [호출] : 클라이언트
# [인자] : w, rij_list, g, q
# [리턴] : commitments (verify를 위한 commitment list)
def generate_commitments(w, rij_list, g, q):
    commitments = []
    for i, rij in enumerate(rij_list):
        if i == 0:
            c = []
            for k in w:
                c.append(g ** k)
            commitments.append(c)
            continue
        commitments.append(np.array(g ** rij, dtype = np.float64))
    
    return commitments


# verify commitments
# [호출] : 클라이언트
# [인자] : g, share(i 에게서 받은 share), commitments(i 가 생성한 commitment list), theta(자신의 theta), q
# [리턴] : boolean
def verify(g, share, commitments, theta, q):
    x = g ** np.array(share) # % q
    
    y = 1
    for i, c in enumerate(commitments):
        if i == 0:
            y = y * (np.array(c)**(theta**i))
        else:
            y = y * (c**(theta**i))

    for i in range(len(x)):
        if np.allclose(x[i], y[i]) == True:
            result = True
        else:
            result = False
            break
    
    return result


# [호출] : 클라이언트
# [인자] : share1, share2(사용자 1과 사용자 2의 거리를 계산하기 위해 1에게 받은 share와 2에게 받은 share를 인자로)
# [리턴] : distance(계산한 거리)
def calculate_distance(shares1, shares2):
    distance = abs(np.array(shares1) - np.array(shares2)) ** 2
    return distance

#[호출] : 서버
#[인자] : theta(theta_list), distances(djk_list)
#[리턴] : _djk(hjk(0))
def generate_h_polynomial(theta, distances):
    f = generateLagrangePolynomial(theta, distances)
    djk = np.polyval(f,0)
    return djk

#[호출] : 서버
#[인자] : _djk(hjk(0)), p, g(처음에 지정해준 p, g)
#[리턴] : 실수 djk
def real_djk(_djk, p, q):
    if(_djk >= ((p-1)/2) and _djk < p):
       _djk = _djk - p
    djk = _djk / (q ** 2)
    return djk

#[호출] : 서버
#[인자] : djk (실수 djk)
#[리턴] : sjk
def calcutate_skj(djk):
    skj = 0
    for i in range(djk):
        skj += djk
    return skj

#[호출] : 서버
#[인자] : skj
#[리턴] : 선택된 유저의 skj, 선택된 유저의 인덱스 값
def select_user(skj):
    tmp = skj[0]
    user = 0
    for i in range(skj):
        if(tmp > skj[i]):
            tmp = skj[i]
            user = i
    return skj[user], user

if __name__ == "__main__":

    q = 7
    g = 3
    n = 4 # N = 40
    T = 3 # T = 7

    model = fl.setup()
    my_model = fl.get_user_dataset(n)

    local_model, local_weight, local_loss = fl.local_update(model, my_model[0], 0)
    model_weights_list = mhelper.weights_to_dic_of_list(local_weight)
    weights_info, flatten_weights = mhelper.flatten_list(model_weights_list)
    
    bar_w = stochasticQuantization(np.array(flatten_weights), g, q)
    
    theta_list = make_theta(n, q)
    rij_list1 = generate_rij(T, q)

    rij_list2 = generate_rij(T, q)
    shares2 = make_shares(bar_w, theta_list, T, rij_list2)
    #commitments1 = generate_commitments(bar_w1, rij_list1, g, q)
    commitments2 = generate_commitments(bar_w, rij_list2, g, q)

    result = verify(g, shares2[0], commitments2, theta_list[0], q)
    print("result: ", result)

    distance = calculate_distance(shares2[0], shares2[1])
    print("distance: ", distance)

    # print(generate_h_polynomial([0,1,2],[1,2,3]))


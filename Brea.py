import random
import numpy as np
import learning.federated_main as fl
import BasicSA as bs
import learning.models_helper as mhelper
from BasicSA import stochasticQuantization
from Turbo import generateLagrangePolynomial

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

p = 7  # modulo p
q = 3  # quantization level q
bar_w = {}


# make N theta_i
# [호출] : 서버
# [인자] : n(사용자수)
# [리턴] : th(각 client들에 대한 theta list)
def make_theta(n):
    global p

    th = []
    for i in range(n):
        th.append(random.randint(1, p))
    return th


# Secret poly coefficients r_ij
# [호출] : 클라이언트
# [인자] : T(threshold)
# [리턴] : rij(사용자 i가 랜덤 생성한 계수)(j: 1~T)
def generate_rij(T):
    global p

    rij = [0, ] # not using first index (0)
    for j in range(T):
        rij.append(random.randint(1, p))
    return rij


# share를 생성하기 위한 다항식 f
# [인자] : theta(자신의 theta), w(weights. quantization한 값), T(threshold), rij(get_rij()의 리턴값)
# [리턴] : y
def f(theta, w, T, rij):
    y = w
    for j in range(1, T+1):
        y = y + (rij[j] * (mod(theta, j, q)))
    return y


# make shares
# [호출] : 클라이언트
# [인자] : flatten_weights, theta_list(서버가 알려준 theta list), T, rij_list
# [리턴] : shares (다른 사용자들에게 보낼 share list)
def make_shares(flatten_weights, theta_list, T, rij_list):
    global bar_w
    shares = []

    # stochastic quantization
    bar_w = stochasticQuantization(np.array(flatten_weights), q, p)

    for theta in theta_list:
        sij = []
        for k in bar_w:
            sij.append(f(theta, k, T, rij_list))
        shares.append(sij)
    return shares


# make commitment
# [호출] : 클라이언트
# [인자] : w, rij_list
# [리턴] : commitments (verify를 위한 commitment list)
def generate_commitments(rij_list):
    global q, p

    commitments = []
    for i, rij in enumerate(rij_list):
        if i == 0:
            c = []
            for k in bar_w:
                c.append(q ** k)
            commitments.append(c)
            continue
        commitments.append(np.array(q ** rij, dtype = np.float64))
    
    return commitments


# verify commitments
# [호출] : 클라이언트
# [인자] : share(i 에게서 받은 share), commitments(i 가 생성한 commitment list), theta(자신의 theta)
# [리턴] : boolean
def verify(share, commitments, theta):
    global p, q
    res = False

    x = q ** np.array(share)  # % q
    y = 1

    for i in commitments:
        for k, c in enumerate(i):
            if k == 0:
                m = mod(theta, k, q)
                y = y * mod(np.array(c), m, q)
            else:
                y = y * (c ** (theta ** k))
        for j in range(len(x)):
            if np.allclose(x[j], y[j]):
                res = True
            else:
                res = False
                break
    return res

def mod(theta, i, q):
    ret = 1
    for idx in range(i):
        ret = ret * theta % q 
    return ret


# [호출] : 클라이언트
# [인자] : shares(sji), n
# [리턴] : distance(계산한 거리)
# def calculate_distance(shares, n, u):
#     distances = []
#     for j in range(n):
#         for k in range(n):
#             dis = abs(np.array(shares[j]) - np.array(shares[k])) ** 2
#             distances.append((u, j, k, dis))
#     return distances

def calculate_distance(shares, n):
    distances = {}
    for j in range(n):
        distances[j] = {}
        for k in range(n):
            dis = abs(np.array(shares[j]) - np.array(shares[k])) ** 2
            distances[j][k] = dis
    return distances

#[호출] : 서버
#[인자] : theta(theta_list), distances(djk_list)
#[리턴] : _djk(hjk(0))
def calculate_djk_from_h_polynomial(theta, distances):
    h = generateLagrangePolynomial(theta, distances)
    djk = np.polyval(h, 0)
    return djk

#[호출] : 서버
#[인자] : _djk(hjk(0)), p, g(처음에 지정해준 p, g)
#[리턴] : 실수 djk
def real_domain_djk(_djk):
    global p, q

    if ((p-1)/2) <= _djk < p:
        _djk = _djk - p
    djk = _djk / (q ** 2)
    return djk

def multi_krum(n, m, djk):
    """
    n = All user
    m = number of selecting user
    djk = distances between users  --> dictionary 
    a = Reed Solomon max number of error
    Sk = selected index set S(k)
    _set = range of adding dju
    skj = list of added dju for each users 
    dis = temporary copy array for one row in skj
    user = selected user's index
    """
    k = 1
    a = (n - k) / 2     # ?
    Sk = []
    while True:
        _set = (n - k + 1) - a - 2
        skj = [0] * (n - k + 1)

        for i in range(len(djk)):
            sum_dis = 0
            for idx, val in djk[i]:
                if idx not in Sk:
                    sum_dis += sum(val)
            skj[i] = sum_dis

        tmp = min(skj)
        index = skj.index(tmp)

        Sk.append(index)
        if k == m:
            break
        k += 1

        for e in range(len(djk)):
            djk.pop(index)

    return Sk

def aggregate_share(shares, selected_user, u):
    si = [0, ]
    for i in selected_user:
        if i != u:
            si = np.array(shares[i]) + si
    return si


def update_weight(_wj, model):
    """
    _wj = weight from user
    demap_wj = wj with demapping function
    model =  global model
    para = paramater using leaning rate and q
    """
    global p, q

    demap_wj = np.array(_wj)
    _model = np.array(model)

    learning_rate = 1
    para = (learning_rate / q)

    for idx_i, val_i in enumerate(demap_wj):
        for idx_j, val_j in enumerate(val_i):
            if ((p - 1) / 2) <= val_j < p:
                demap_wj[idx_i][idx_j] = para * (val_j - p)
            else:
                demap_wj[idx_i][idx_j] = para * val_j

    return _model - demap_wj


if __name__ == "__main__":
    n = 4 # N = 40
    T = 3 # T = 7



    model = fl.setup()
    my_model = fl.get_user_dataset(n)

    local_model, local_weight, local_loss = fl.local_update(model, my_model[0], 0)
    model_weights_list = mhelper.weights_to_dic_of_list(local_weight)
    weights_info, flatten_weights = mhelper.flatten_list(model_weights_list)

    theta_list = make_theta(n)
    rij_list1 = generate_rij(T)

    rij_list2 = generate_rij(T)
    shares2 = make_shares(flatten_weights, theta_list, T, rij_list2)
    # commitments1 = generate_commitments(bar_w1, rij_list1, g, q)

    commitments2 = [generate_commitments(rij_list2)]
    for i in range(n-1):
        commitments2.append(commitments2[0])

    # 0 user verify
    result = verify(shares2[0], commitments2, theta_list[0])

    print(theta_list)
    print(rij_list2)
    print(shares2)
    print(commitments2)
    print(result)
    print(aggregate_share(shares2, [2,3], 1))
    distance = calculate_distance(shares2, 4)
    print("distance: ", distance)
    print(multi_krum(4, 3,distance))
    # print(calculate_djk_from_h_polynomial(theta_list, distance))


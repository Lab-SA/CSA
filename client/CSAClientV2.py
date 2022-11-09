import os, sys, json, random, socket

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import CSA
from CSAClient import CSAClient
from CommonValue import CSARound
from BasicSA import stochasticQuantization
from common import sendRequestAndReceiveV2, sendRequestV2
import learning.federated_main as fl
import learning.models_helper as mhelper
from dto.CSASetupDto import CSASetupDto
from ast import literal_eval

SIZE = 2048
ENCODING = 'utf-8'

class CSAClientV2(CSAClient):
    def setUp(self):
        tag = CSARound.SetUp.name
        print("Start New Round")

        self.my_sk, self.my_pk = CSA.generateECCKey()
        self.PS = random.randrange(0, 3)
        self.GPS_i = random.randrange(1, 6)
        self.GPS_j = random.randrange(1, 8)

        # request with my public key (pk)
        # response: CSASetupDto
        response = sendRequestAndReceiveV2(self.mysocket, tag, {'pk': self.my_pk.hex(), 'PS': self.PS, 'GPS_i': self.GPS_i, 'GPS_j': self.GPS_j})
        setupDto = json.loads(json.dumps(response), object_hook=lambda d: CSASetupDto(**d))

        self.n = setupDto.n
        self.g = setupDto.g
        self.p = setupDto.p
        self.R = setupDto.R
        self.cluster = setupDto.cluster
        self.clusterN = setupDto.clusterN # deprecated
        self.cluster_indexes = setupDto.cluster_indexes
        self.index = setupDto.index
        self.quantizationLevel = setupDto.qLevel
        self.others_keys = {int(key): value for key, value in literal_eval(setupDto.cluster_keys).items()}
        self.others_keys.pop(self.index)
        self.training_weight = setupDto.training_weight
        #print(self.training_weight)

        self.cluster_indexes.remove(self.index)

        # decrypt ri and verity Ri = (g ** ri) mod p
        self.ri = int(bytes.decode(CSA.decrypt(self.my_sk, bytes.fromhex(setupDto.encrypted_ri)), 'ascii'))
        self.Ri = int(setupDto.Ri)
        if self.Ri != (self.g ** self.ri) % self.p:
            print("Invalid Ri and ri.")

        # print(self.cluster, self.clusterN, self.index, self.ri, self.Ri)
        # print(self.others_keys)

        self.data = setupDto.data
        global_weights = mhelper.dic_of_list_to_weights(literal_eval(setupDto.weights))
        #print(self.cluster, len(self.data))

        #self.weight = [1] # temp
        if self.model == {}:
            self.model = fl.setup()
        fl.update_model(self.model, global_weights)
        local_model, local_weight, local_loss = fl.local_update(self.model, self.data, 0) # epoch 0 (temp)
        self.weights_info, self.weight = mhelper.flatten_tensor(local_weight)
        self.weight = list(self.training_weight * x for x in self.weight)
        self.weight = stochasticQuantization(self.weight, self.quantizationLevel, self.p)
        #print(self.index, self.weight[250:260])


    def shareRandomMasks(self): # send random masks
        tag = CSARound.ShareMasks.name

        while True: # repete until all member share valid masks
            self.my_mask, encrypted_mask, public_mask = CSA.generateMasks(self.index, self.cluster_indexes, self.ri, self.others_keys, self.g, self.p)
            request = {'cluster': self.cluster, 'index': self.index, 'emask': encrypted_mask, 'pmask': public_mask}
            response = sendRequestAndReceiveV2(self.mysocket, tag, request)
            # print(self.my_mask, public_mask)
            if response.get('process') is not None: # this cluster is end
                return False

            prev_survived = len(self.cluster_indexes) + 1 # include my index
            if len(response['survived']) != prev_survived: # drop out in shareRandomMasks stage!
                self.cluster_indexes = response['survived']
                self.cluster_indexes.remove(self.index)
                continue

            emask = {int(key): value for key, value in response['emask'].items()}
            pmask = {int(key): value for key, value in response['pmask'].items()}
            # print(self.ri, self.Ri, self.index)

            self.nowN = len(pmask)
            self.others_mask = CSA.verifyMasks(self.index, self.ri, emask, pmask, self.my_sk, self.g, self.p)
            #print('my mask', self.my_mask)
            #print('others mask', self.others_mask)
            if self.others_mask != {}:
                request = {'cluster': self.cluster, 'index': self.index}
                sendRequestV2(self.mysocket, self.verifyRound, request)
                break
        return True

    def sendSecureWeight(self): # send secure weight S
        tag = CSARound.Aggregation.name

        if self.isBasic:    # BasicCSA
            self.a = 0
        else:               # FullCSA
            self.a = random.randrange(1, self.p)
        S = CSA.generateSecureWeight(self.weight, self.ri, self.others_mask, self.p, self.a)
        #print(S[250:260])
        request = {'cluster': self.cluster, 'index': self.index, 'S': S}
        response = sendRequestAndReceiveV2(self.mysocket, tag, request)

        self.survived = response['survived']
        if not self.isBasic or len(self.survived) != self.nowN:
            self.sendMasksOfDropout()

    def sendMasksOfDropout(self): # send the masks of drop-out users
        tag = CSARound.RemoveMasks.name

        request = {'cluster': self.cluster, 'index': self.index}
        if not self.isBasic:
            request['a'] = self.a
        while True:
            RS = CSA.computeReconstructionValue(self.survived, self.my_mask, self.others_mask, self.cluster_indexes)
            request['RS'] = RS
            response = sendRequestAndReceiveV2(self.mysocket, tag, request)
            self.survived = response['survived']
            if len(self.survived) == 0:
                break

if __name__ == "__main__":
    client = CSAClientV2(isBasic = True) # BasicCSA
    for _ in range(3):
        client.setUp()
        if not client.shareRandomMasks():
            continue
        client.sendSecureWeight()

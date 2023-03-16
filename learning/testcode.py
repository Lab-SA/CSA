import copy

from federated_main import setup, get_user_dataset, local_update, test_model
from utils import sum_weights, average_weight
from common import writeWeightsToFile,writeToExcel, readWeightsFromFile
import learning.models_helper as mhelper

if __name__ == "__main__":
    # test federated learning

    # setup model with base weights
    global_model = setup()

    model_shape = mhelper.get_model_weights(global_model)
    print(model_shape)
    weights_info, flatten_weights = mhelper.flatten_tensor(global_model.state_dict())
    print(weights_info)  # print default_weights_info
    writeWeightsToFile(flatten_weights)  # write one-dimensional weights info to model.txt
    # use global default_weights_info defined in model_helper.py
    # base_weights = mhelper.restore_weights_tensor(mhelper.default_weights_info, readWeightsFromFile())

    # use weights_info returned by mhelper.flatten_tensor()
    base_weights = mhelper.restore_weights_tensor(weights_info, readWeightsFromFile())
    global_model.load_state_dict(base_weights)

    # get data set
    n = 100
    user_groups = get_user_dataset(n)

    run_data = []

    print(f'\n | TESTING CODE |\n')

    for epoch in range(100):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        for idx in range(n):
            local_model, local_weight, local_loss = local_update(global_model, user_groups[idx], epoch)
            local_weights.append(copy.deepcopy(local_weight))
            #print(local_weight['conv1.bias'])

        # update global weights
        sum_weight = sum_weights(local_weights) # sum
        avg_weight = average_weight(sum_weight, n) # average
        global_model.load_state_dict(avg_weight)  # update
        #print(avg_weight['conv1.bias'])

        # test model
        acc = test_model(global_model)
        run_data.append([epoch+1, acc])

        print(acc)

    writeToExcel('../../results/csa.xlsx', run_data)

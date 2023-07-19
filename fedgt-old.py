import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from numpy.random import seed
# seed(10)
from tensorflow.random import set_seed
# set_seed(20)
# tf.debugging.set_log_device_placement(True)

import math
import gc


def filter_client_data(images, labels):
    clients = 5
    filtered_images = []
    filtered_labels = []
    selector = []
    selector.append( (labels == 0) | (labels == 1) )
    selector.append( (labels == 2) | (labels == 3) )
    selector.append( (labels == 4) | (labels == 5) )
    selector.append( (labels == 6) | (labels == 7) )
    selector.append( (labels == 8) | (labels == 9) )
    # selector.append( (labels == 0) | (labels == 1) | (labels == 2) )
    # selector.append( (labels == 3) | (labels == 4) | (labels == 5) )
    # selector.append( (labels == 6) | (labels == 7) | (labels == 8) | (labels == 9) )
    for s in selector:
        filtered_images.append(
            images[s]
        )
        filtered_labels.append(
            labels[s]
        )
    
    return filtered_images, filtered_labels


def shuffle_client_test_data(images, labels, foreign_data_split):
    clients = len(images)
    x = []
    y = []
    for i in range(clients):
        x.append(
            images[i]
        )
        y.append(
            labels[i]
        )
        for j in range(clients):
            if i == j:
                continue
            x[-1] = np.concatenate(
                ( x[-1], images[j][0:int(foreign_data_split * images[j].shape[0])] ),
                axis=0
            )
            y[-1] = np.concatenate(
                ( y[-1], labels[j][0:int(foreign_data_split * labels[j].shape[0])] ),
                axis=0
            )
    return x, y


def pick_random_sample_test_data(images, labels, split):
    x = []
    y = []
    clients = len(images)
    for i in range(clients):
        indices = np.random.randint( 0, images[i].shape[0], size=int(split*images[i].shape[0]) )
        x.append(
            np.take(images[i], indices, axis=0)
        )
        y.append(
            np.take(labels[i], indices, axis=0)
        )
    return x, y


def get_init_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model


def set_weight_model(model, average_weights):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.set_weights(average_weights)
    return model


def fit_model(model, x, y):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(x, y, epochs=5, batch_size=512)
    return model


def fl_scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def fl_sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


def fed_avg(models, scaling_factor):
    scaled_local_weight_list = list()
    for m in range(len(models)):
        scaled_weights = fl_scale_model_weights(models[m].get_weights(), scaling_factor[m])
        scaled_local_weight_list.append(scaled_weights)
    fed_average_weights = fl_sum_scaled_weights(scaled_local_weight_list)
    return fed_average_weights


def gt_fed_avg(models, x, y):
    gt_models = []
    clients = len(models)
    for i in range(clients):
        accuracy = []
        for j in range(clients):
            accuracy.append(
                models[j].evaluate(x[i], y[i], batch_size=512)[1]
            )
        sum_of_acc = sum(accuracy)
        scaling_factor_strategy = [float(k)/float(sum_of_acc) for k in accuracy]
        gt_models.append(
            set_weight_model(get_init_model(), fed_avg(models, scaling_factor_strategy))
        )
    return gt_models


df_data = {
    "client": [],
    "fedavg_acc": [],
    "gt_fedavg_acc": []
}


def main(run, foreign_data_split, gt_test_data_split):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    print(f"Data downloaded for train:{train_images.shape} and test:{test_images.shape}")

    train_x, train_y = filter_client_data(train_images, train_labels)
    print(f"Train data filtered as localized client data")

    _test_x, _test_y = filter_client_data(test_images, test_labels)
    test_x, test_y = shuffle_client_test_data(np.copy(_test_x), np.copy(_test_y), foreign_data_split)
    del _test_x
    del _test_y

    game_x, game_y = pick_random_sample_test_data(test_x, test_y, gt_test_data_split)
    print(f"Sampling from test data completed for game rounds in game theory federated average")

    local_models = []

    # fed_avg_scaling_factor = [0.3, 0,3, 0.4]
    fed_avg_scaling_factor = [0.2, 0,2, 0.2, 0.2, 0.2]
    

    clients = len(train_x)

    for i in range(clients):
        local_models.append(
            fit_model(get_init_model(), train_x[i], train_y[i])
        )
        print(f"Local model build completed for client={i+1}")

    fed_avg_model = set_weight_model(get_init_model(), fed_avg(local_models, fed_avg_scaling_factor))
    print(f"Federated average aggregation completed")

    gt_fed_avg_models = gt_fed_avg(local_models, game_x, game_y)
    print(f"Game theory based federated average aggregation completed")

    for j in range(clients):
        df_data["client"].append(j+1)

        _, fedavg_acc = fed_avg_model.evaluate(test_x[j], test_y[j], batch_size=512)
        df_data["fedavg_acc"].append(fedavg_acc)

        _, gt_fedavg_acc = gt_fed_avg_models[j].evaluate(test_x[j], test_y[j], batch_size=512)
        df_data["gt_fedavg_acc"].append(gt_fedavg_acc)

        print(f"Evaluation completed for client={j+1}")

    df = pd.DataFrame(df_data)
    df.to_csv(f'run_{run}_fs={foreign_data_split}_gts={gt_test_data_split}.csv', index=False)


if __name__ == "__main__":
    run_id = str(sys.argv[1])
    fs = [0.1, 0.2, 0.3, 0.4, 0.5]
    gts = [0.1, 0.2, 0.3, 0.4, 0.5]
    for f in fs:
        for g in gts:
            main(run_id, f, g)
    
    # main(run_id, 0.1, 0.1)
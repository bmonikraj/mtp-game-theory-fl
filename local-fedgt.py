import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from numpy.random import seed
from tensorflow.random import set_seed

import math
import gc


def filter_client_data(images, labels, K):
    filtered_images = []
    filtered_labels = []
    selector = []

    NC = 100
    Step = int(100/K)
    for i in range(0, NC, Step):
        selector.append(
            np.isin(labels, np.array([j for j in range(i, i+Step)]))
        )

    """
    # MNIST data 
    for s in selector:
        filtered_images.append(
            images[s]
        )
        filtered_labels.append(
            labels[s]
        )
    """

    for s in selector:
        filtered_images.append(
            images[s.reshape(images.shape[0])]
        )
        filtered_labels.append(
            labels[s.reshape(images.shape[0])]
        )
    
    
    return filtered_images, filtered_labels


def filter_homogenous_client_data(images, labels, K):
    filtered_images = []
    filtered_labels = []
    selector = []

    for i in range(K):
        selector.append(
            np.random.choice(len(labels), int(len(labels)/K), replace=False)
        )
    

    for s in selector:
        filtered_images.append(
            images[s]
        )
        filtered_labels.append(
            labels[s]
        )

    return filtered_images, filtered_labels


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
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100, activation="softmax")
    ])

    """
    # (*) MNIST
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    """

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
    model.fit(x, y, epochs=30, batch_size=512)
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


def gt_fed_avg(models, x, y, theta):
    gt_models = []
    clients = len(models)
    for i in range(clients):
        accuracy = []
        selected_models = []
        for j in range(clients):
            acc = models[j].evaluate(x[i], y[i], batch_size=512)[1]
            if acc >= theta:
                accuracy.append(
                    acc
                )
                selected_models.append(
                    models[j]
                )
        # sum_of_acc = sum(accuracy)
        _n_ = len(selected_models)
        scaling_factor_strategy = [float(k)/float(_n_) for k in accuracy]
        print(f"SS GT {scaling_factor_strategy} _n_ {_n_}")
        gt_models.append(
            set_weight_model(get_init_model(), fed_avg(selected_models, scaling_factor_strategy))
        )
    return gt_models


df_data = {
    "run": [],
    "client": [],
    "fedavg_acc": [],
    "gt_fedavg_acc": []
}


def main(run, K, gt_test_data_split, theta, homogenous=False):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    print(f"Data downloaded for train:{train_images.shape} and test:{test_images.shape}")

    if not homogenous:
        train_x, train_y = filter_client_data(train_images, train_labels, K)
        print(f"Train data filtered as localized client data")

        test_x, test_y = filter_client_data(test_images, test_labels, K)
    else:
        train_x, train_y = filter_homogenous_client_data(train_images, train_labels, K)
        print(f"Train data filtered as localized client data")

        test_x, test_y = filter_homogenous_client_data(test_images, test_labels, K)

    game_x, game_y = pick_random_sample_test_data(test_x, test_y, gt_test_data_split)
    print(f"Sampling from test data completed for game rounds in game theory federated average")

    local_models = []

    fed_avg_scaling_factor = [float(1/K) for i in range(K)]

    clients = len(train_x)

    for i in range(clients):
        local_models.append(
            fit_model(get_init_model(), train_x[i], train_y[i])
        )
        print(f"Local model build completed for client={i+1}")

    fed_avg_model = set_weight_model(get_init_model(), fed_avg(local_models, fed_avg_scaling_factor))
    print(f"Federated average aggregation completed")

    gt_fed_avg_models = gt_fed_avg(local_models, game_x, game_y, theta)
    print(f"Game theory based federated average aggregation completed")

    for j in range(clients):
        df_data["run"].append(run)

        df_data["client"].append(j+1)

        _, fedavg_acc = fed_avg_model.evaluate(test_x[j], test_y[j], batch_size=512)
        df_data["fedavg_acc"].append(fedavg_acc)

        _, gt_fedavg_acc = gt_fed_avg_models[j].evaluate(test_x[j], test_y[j], batch_size=512)
        df_data["gt_fedavg_acc"].append(gt_fedavg_acc)

        print(f"Evaluation completed for client={j+1}")


if __name__ == "__main__":
    runs = 5
    K = int(sys.argv[1])
    gts = float(sys.argv[2])
    theta = float(sys.argv[3])
    homogenous = int(sys.argv[4])
    homogenity = {
        0: False,
        1: True
    }
    for run_id in range(1,runs+1):
        main(run_id, K, gts, theta, homogenous=homogenity[homogenous])

    df = pd.DataFrame(df_data)
    agg_df = df.groupby('client', as_index=False).mean()[['fedavg_acc', 'gt_fedavg_acc']]
    agg_df  = agg_df.astype(float)
    agg_df.to_csv(f'K={K}_homogenity={str(homogenity[homogenous])}_theta={theta}.csv', index=False)

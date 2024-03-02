import h5py
import torch
import numpy as np
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root="../data", train=True, download=True, transform=tensor_transform
    )
    test_set = datasets.MNIST(
        root="../data", train=False, download=True, transform=tensor_transform
    )

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)

    return x_train, y_train, x_test, y_test

def load_twomoon() -> tuple:
    data, y = make_moons(n_samples=7000, shuffle=True, noise=0.075, random_state=None)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, test_size=0.33, random_state=42
    )
    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    return x_train, y_train, x_test, y_test


def load_reuters() -> tuple:
    with h5py.File("../data/Reuters/reutersidf_total.h5", "r") as f:
        x = np.asarray(f.get("data"), dtype="float32")
        y = np.asarray(f.get("labels"), dtype="float32")

        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def load_iris() -> tuple:
    import sklearn.datasets as datasets
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.10
    )
    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    
    return x_train, y_train, x_test, y_test


def load_from_openml(dataset_id: int) -> tuple:
    x, y = fetch_openml(data_home="../data", data_id=dataset_id, as_frame=False, return_X_y=True)
    y = y.astype(np.int32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.10
    )
    if dataset_id == 1478: # har, eeg
        y_train, y_test = y_train - 1, y_test - 1

    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    
   
    return x_train, y_train, x_test, y_test


def load_data(dataset: str, reduced: bool = False) -> tuple:
    """
    This function loads the dataset specified in the config file.


    Args:
        dataset (str or dictionary):    In case you want to load your own dataset,
                                        you should specify the path to the data (and label if applicable)
                                        files in the config file in a dictionary fashion under the key "dataset".

    Raises:
        ValueError: If the dataset is not found in the config file.

    Returns:
        tuple: A tuple containing the train and test data and labels.
    """

    if dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == "twomoons":
        x_train, y_train, x_test, y_test = load_twomoon()
    elif dataset == "reuters":
        x_train, y_train, x_test, y_test = load_reuters()
    elif dataset == "iris":
        x_train, y_train, x_test, y_test = load_iris()
    else:
        x_train, y_train, x_test, y_test = load_from_openml(int(dataset))

    return x_train, x_test, y_train, y_test

import os
import pickle

import torch.utils.data as data
from litsr.utils import load_file_list
from scipy import io as scio
from tqdm import tqdm


def load_mat(path):
    mat = scio.loadmat(path, verify_compressed_data_integrity=False)
    return mat["t"]


class HSImageFolder(data.Dataset):
    """
    Construct a hyper-spectral image dataset from a folder
    """

    def __init__(self, datapath, repeat=1, cache=None, first_k=None):
        self.datapath = datapath
        self.repeat = repeat or 1
        self.cache = cache

        if cache == "bin":
            [self.filenames, self.files] = self.load_data_from_bin(first_k)
        elif cache == "memory":
            [self.filenames, self.files] = self.load_data_from_memery(first_k)
        else:
            [self.filenames, self.files] = self.load_data()

        if first_k:
            self.filenames = self.filenames[:first_k]
            self.files = self.files[:first_k]

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "memory":
            return x
        elif self.cache == "bin":
            with open(x, "rb") as f:
                x = pickle.load(f)
            return x
        else:
            return load_mat(x)

    def load_data_from_bin(self, first_k=None):
        file_names = load_file_list(self.datapath, regx=".*.{0}".format(self.format))
        file_names = [os.path.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        bin_root = os.path.join(
            os.path.dirname(self.datapath), "_bin_" + os.path.basename(self.datapath)
        )
        if not os.path.exists(bin_root):
            os.mkdir(bin_root)
            print("mkdir", bin_root)
        files = []

        for f in file_names:
            bin_file = os.path.join(bin_root, f.split(".")[0] + ".pkl")
            if not os.path.exists(bin_file):
                with open(bin_file, "wb") as bin_f:
                    pickle.dump(
                        load_mat(os.path.join(self.datapath, f)), bin_f,
                    )
                print("dump", bin_file)
            files.append(bin_file)
        return file_names, files

    def load_data_from_memery(self, first_k=None, from_pickle=True):
        if from_pickle:
            file_names, filenames = self.load_data_from_bin(first_k)
            files = []
            pbar = tqdm(filenames)
            pbar.set_description("load data (from pickle)")
            for f in pbar:
                with open(f, "rb") as f_:
                    files.append(pickle.load(f_))

            return file_names, files

        file_names = load_file_list(self.datapath, regx=".*.{0}".format(self.format))
        file_names = [os.path.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        files = []
        pbar = tqdm(file_names)
        pbar.set_description("load data")
        for f in pbar:
            files.append(load_mat(os.path.join(self.datapath, f)))

        return file_names, files

    def load_data(self):
        files = load_file_list(self.datapath, regx=".*.{0}".format(self.format))
        file_names = [os.path.basename(n) for n in files]
        return file_names, files

    def __len__(self):
        return len(self.files) * self.repeat


class PairedImageFolder(data.Dataset):
    def __init__(self, path1, path2, repeat=1, cache=None, first_k=None):
        self.dataset_1 = HSImageFolder(
            path1, repeat=repeat, cache=cache, first_k=first_k
        )
        self.dataset_2 = HSImageFolder(
            path2, repeat=repeat, cache=cache, first_k=first_k
        )

        self.filenames = self.dataset_2.filenames

        assert len(self.dataset_1) == len(self.dataset_2)

    def __getitem__(self, i):
        return tuple([self.dataset_1[i], self.dataset_2[i]])

    def __len__(self):
        return len(self.dataset_1)

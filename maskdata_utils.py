import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
from matplotlib import pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split


class FaceMaskHandler(object):
    ''' Provides a convenient interface to manipulate face mask data '''

    def __init__(self, img_size = 128, train_ratio = 1.0):
        # Download data if needed
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

    def load_dataset(self):
        directory = 'resources/dataset/'
        with_mask_list = sorted(glob.glob(directory + '/with_mask/*'))#[:50]
        without_mask_list = sorted(glob.glob(directory + '/without_mask/*'))#[:50]
        
        img_list = with_mask_list + without_mask_list
        X = np.stack([np.array(Image.open(f).resize((self.img_size, self.img_size)))  for f in img_list])
        X_scaled = X / 255.
        y = np.array([1]*len(with_mask_list) + [0]*len(without_mask_list))
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        X_train, X_val = X_train[:-500], X_train[-500:]
        y_train, y_val = y_train[:-500], y_train[-500:]

        n = int(self.train_ratio * len(X_train))
        X_train = X_train[:n]
        y_train = y_train[:n]

        print(f'train: {len(X_train)}, validation: {X_val.shape[0]}, test:{X_test.shape[0]}')

        return X_train, y_train, X_val, y_val, X_test, y_test


    def process_batch(self, batch, batch_size, color=False, rescale=True):
        # Rescale to range [-1, +1]
        if rescale:
            batch = batch * 2 - 1

        return batch

    def get_batch(self, subset, batch_size, color=False, rescale=True):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)

        # Process batch
        batch = self.process_batch(X[idx], batch_size, color, rescale)

        # Image label
        labels = y[idx]

        return batch.astype('float32'), labels.astype('int32')

    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len



class FaceMaskGenerator(object):

    ''' Data generator providing face mask data '''

    def __init__(self, batch_size, subset, train_ratio = 1.0, color=False, rescale=True):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.fackmask_handler = FaceMaskHandler(train_ratio = train_ratio)
        self.n_samples = self.fackmask_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        # Get data
        x, y = self.fackmask_handler.get_batch(self.subset, self.batch_size, rescale=self.rescale)

        # Convert y to one-hot
        y_h = np.eye(2)[y]

        tile_size = 32
        n = int((128 // (tile_size / 2)) - 1)
        
        arr = []
        for i in range(n):
            for j in range(n):
                arr.append(x[:, i*16:i*16+32, j*16:j*16+32, :])
        images = np.array(arr).transpose((1, 0, 2, 3, 4))
        
        return images, y_h



class FaceMaskTileGenerator(object):

    ''' Data generator providing tiles of face mask images '''

    def __init__(self, batch_size, subset, positive_samples=1, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = 3
        self.batch_size = batch_size
        self.subset = subset
        self.terms = 4
        self.rescale = rescale

        # Initialize MNIST dataset
        self.fackmask_handler = FaceMaskHandler(136)
        self.n_samples = self.fackmask_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):
        tile_size = 32
        ts = 34
        hs = int(ts / 2)
        x, _ = self.fackmask_handler.get_batch(self.subset, self.batch_size, rescale=self.rescale)
        n = int((x.shape[1] // (tile_size / 2)) - 1)

        # divide into tiles
        arr = []
        blocks = []
        for i in range(n):
            for j in range(n):
                tile = x[:, i*hs:i*hs+ts, j*hs:j*hs+ts, :]
                blocks.append(tile)
                # data augmentation: random crop
                dx, dy = np.random.randint(3, size=2)
                tile = tile[:, dx:dx+tile_size, dy:dy+tile_size, :]

                # data augmentation: horizontal flip
                if np.random.randint(2) == 0:
                    tile = tile[:, :,::-1, :]

                arr.append(tile)
        
        blocks = np.array(blocks).transpose((1, 0, 2, 3, 4))
        blocks = blocks.reshape((self.batch_size * n, n, ts, ts, 3))

        images = np.array(arr).transpose((1, 0, 2, 3, 4))
        images = images.reshape((self.batch_size * n, n, tile_size, tile_size, 3))


        # Build sentences
        sentence_labels = np.ones((self.batch_size * n, 1)).astype('int32')
        positive_samples_n = self.positive_samples * n
        sentence_labels[:positive_samples_n,0] = 0

        nl = []
        for k in range(self.predict_terms):
            nlist = np.arange(0, n)
            nlist = nlist[nlist != (n-self.predict_terms+k)]
            nl.append(nlist)

        idx = range(positive_samples_n, self.batch_size * n)
        for k in range(self.predict_terms):
            tiles = []
            for b in idx:
                nrri = np.random.choice(nl[k])
                tiles.append(blocks[b, nrri, :])

            tile = np.stack(tiles)
            # data augmentation: random crop
            dx, dy = np.random.randint(3, size=2)
            tile = tile[:, dx:dx+tile_size, dy:dy+tile_size, :]

            # data augmentation: horizontal flip
            if np.random.randint(2) == 0:
                tile = tile[:, :,::-1, :]

            images[idx, n-self.predict_terms+k, :] = tile#images[b, nrri, :]


        # for b in range(positive_samples_n, self.batch_size * n):
        #     for k in range(self.predict_terms):
        #         nrri = np.random.choice(nl[k])

        #         tile = blocks[b, nrri, :]

        #         # data augmentation: random crop
        #         dx, dy = np.random.randint(3, size=2)
        #         tile = tile[dx:dx+tile_size, dy:dy+tile_size, :]

        #         # data augmentation: horizontal flip
        #         if np.random.randint(2) == 0:
        #             tile = tile[:,::-1, :]

        #         images[b, n-self.predict_terms+k, :] = tile#images[b, nrri, :]


        # Retrieve actual images
        # images, _ = self.mnist_handler.get_batch_by_labels(self.subset, image_labels.flatten(), self.image_size, self.color, self.rescale)

        # Assemble batch
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]



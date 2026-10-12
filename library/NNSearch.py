import math
import random
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from pynndescent import NNDescent     # <-- NEW
from library.timing import timing



def randomIndices(size, outputSize=None, indicesToIgnore=None):
    indices = list(range(size))

    if indicesToIgnore is not None:
        for x in indicesToIgnore:
            indices.remove(x)

    size = len(indices)
    if outputSize is None or outputSize > size:
        outputSize = size

    r = []
    for _ in range(outputSize):
        size -= 1
        if size < 0:
            break
        if size == 0:
            r.append(indices[0])
        else:
            p = random.randint(0, size)
            x = indices[p]
            r.append(x)
            indices.remove(x)
    
    return r



class NNSearch:
    def __init__(self, nebSize=5, timingDict=None):
        self.nebSize = nebSize
        self.neighbourhoods = []
        self.timingDict = timingDict
        self.basePoints = []
        self.index = None



    def timerStart(self, name):
        if self.timingDict is not None:
            if name not in self.timingDict:
                self.timingDict[name] = timing(name)
            self.timingDict[name].start()

    def timerStop(self, name):
        if self.timingDict is not None:
            if name in self.timingDict:
                self.timingDict[name].stop()



    def neighbourhoodOfItem(self, i):
        return self.neighbourhoods[i]

    def getNbhPointsOfItem(self, index):
        return self.getPointsFromIndices(self.neighbourhoodOfItem(index))

    def getPointsFromIndices(self, indices):
        permutation = randomIndices(len(indices))
        nmbi = np.array(indices)[permutation]
        nmb = self.basePoints[nmbi]
        return tf.convert_to_tensor(nmb)



    def neighbourhoodOfItemList(self, items, maxCount=None):
        nbhIndices = set()
        duplicates = []
        for i in items:
            for x in self.neighbourhoodOfItem(i):
                if x in nbhIndices:
                    duplicates.append(x)
                else:
                    nbhIndices.add(x)

        nbhIndices = list(nbhIndices)
        if maxCount is not None:
            if len(nbhIndices) < maxCount:
                nbhIndices.extend(duplicates)
            nbhIndices = nbhIndices[0:maxCount]

        return self.getPointsFromIndices(nbhIndices)



    # --------------------- NN-DESCENT VERSION ---------------------
    def fit(self, haystack, needles=None, nebSize=None):
        self.timerStart("NN_fit_chained_init")

        if nebSize is None:
            nebSize = self.nebSize
        if needles is None:
            needles = haystack

        self.basePoints = haystack.astype(np.float32)

        # Build NN-Descent index
        index = NNDescent(
            self.basePoints,
            n_neighbors=nebSize,
            metric="euclidean",
            n_jobs=-1,          # full CPU parallelism
            low_memory=True
        )
        index.prepare()
        self.index = index

        self.timerStop("NN_fit_chained_init")
        self.timerStart("NN_fit_chained_toList")

        # Batch querying (much faster than looping)
        indices, distances = index.query(
            needles.astype(np.float32),
            k=nebSize
        )

        # Convert to your internal format
        self.neighbourhoods = [ind for ind in indices]

        self.timerStop("NN_fit_chained_toList")
        return self
    # --------------------------------------------------------------

"""

"""
import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import sys
import time
import asyncio
import pickle
import logging 

#logging.basicConfig()
log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

from tensorflow.python.keras.layers import InputSpec
from policy import Policy
np.set_printoptions(linewidth=np.nan, suppress=True)

COMMON_HIDDEN_LAYER_SIZES = [80, 60, 40]
OUTPUT_HIDDEN_LAYER_SIZES = [40, 40]

#Only part of the input vector needs embedding, 
#so use mask layer to filter out the non-embedding part
class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, mask, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"))
        super().__init__(**kwargs)
        self.mask = mask
        self.input_spec = InputSpec(ndim=2)
        self.units = np.sum(self.mask)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == self.mask.shape[0]
        super().build(input_shape)
    
    def call(self, invalues):
        return tf.reshape(tf.boolean_mask(invalues, self.mask, axis=1), [-1, np.sum(self.mask)])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

#Rescale the predicted output to float between 0 and 1 before feeding it to next level network
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, numin, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"))
        super().__init__(**kwargs)
        self.numin = numin
        self.input_spec = InputSpec(ndim=2)
        self.units = 1

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] == self.numin
        ramp = np.linspace(0, 1, self.numin, False, dtype=np.float32)
        self.scale = tf.constant(ramp, shape=[self.numin, 1], name="ramp")

        super().build(input_shape)
    
    def call(self, invalues):
        return tf.matmul(invalues, self.scale)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# Embedding the input state according to modtype.stateembed
class StateEmbedModel(tf.keras.models.Model):
    def __init__(self, statedim, kerneldims, stateembed, trainable=True):
        super().__init__()
        self.statedim = statedim
        self.kerneldims = kerneldims
        self.stateembed = stateembed
        self.emblayers = []
        for embeddim in self.kerneldims:
            self.emblayers.append(tf.keras.layers.Dense(embeddim[1], 
                                                        use_bias=False,
                                                        trainable=trainable))
        self.restmask = np.full((self.statedim,), True)
        self.emboutsize = 0
        for svec in self.stateembed:
            start = svec[0]
            stop = svec[1]
            for kv in range(svec[3]):
                self.emboutsize += self.emblayers[svec[2]].units
                self.restmask[start:stop] = False
                start += svec[4]
                stop += svec[4]
        self.masklayer = MaskLayer(self.restmask)
        self.units = self.emboutsize + self.masklayer.units

    def call(self, inputs):
        embout = []
        for svec in self.stateembed:
            start = svec[0]
            stop = svec[1]
            for kv in range(svec[3]):
                embout.append(self.emblayers[svec[2]](tf.slice(inputs, [0, start], [-1, stop - start])))
                start += svec[4]
                stop += svec[4]
        rest = self.masklayer(inputs)
        return tf.concat(embout + [rest], 1)

#Build policy network
class PolicyModel(tf.keras.models.Model):
    def __init__(self, Pi_insize, Pi_outsize, hiddensize, trainable=True):
        super().__init__()
        self.Pi_insize = Pi_insize
        self.Pi_outsize = Pi_outsize
        self.units = self.Pi_outsize
        self.hiddensize = hiddensize
        self.hiddenlayers = []
        for size in self.hiddensize:
            self.hiddenlayers.append(tf.keras.layers.Dense(size, activation=tf.nn.selu, trainable=trainable))
        self.Pi_output = tf.keras.layers.Dense(Pi_outsize, activation=tf.nn.log_softmax, trainable=trainable)

    def call(self, inputs):
        layers = [inputs]
        for h in self.hiddenlayers:
            layers.append(h(layers[-1]))
        layers.append(self.Pi_output(layers[-1]))
        return layers

#Build all policy networks
class PolicyPAPPO(Policy):
    algorithm = "PAPPO"

    class Model(tf.keras.models.Model):
        def __init__(self, modtype, trainable=True):
            super().__init__()
            self.statedim = modtype.statedim
            self.actdim = modtype.actdim
            self.kerneldims = modtype.embedkernels
            self.stateembed = modtype.stateembed
            self.actslices = modtype.actslices
            self.slicedims = modtype.slicedims
            self.givens = modtype.givens
            self.embedlayer = StateEmbedModel(self.statedim, self.kerneldims, self.stateembed, trainable=trainable)

            self.Pi = []
            self.Pi_outlayers = []
            self.Pi_nextoutsizes = []
            self.actscales = {}
            for act in range(len(self.actslices)):
                if self.actslices[act][2] is 256:
                    self.actscales[act] = ScaleLayer(self.slicedims[act][0])
            for pi in range(len(self.slicedims)):
                if pi is 0:
                    Pi_insize = self.embedlayer.units
                else:
                    Pi_insize = COMMON_HIDDEN_LAYER_SIZES[-1]

                Pi_outsize = self.slicedims[pi][0]
                addins = self.givens[pi]
                for act in addins:
                    if act >= 0:
                        Pi_insize += self.Pi_nextoutsizes[act]
                if self.actslices[pi][2] is 255:
                    Pi_outlayer = None
                elif self.actslices[pi][2] is 256:
                    Pi_outlayer = self.actscales[pi]
                else:
                    Pi_outlayer = self.embedlayer.emblayers[self.actslices[pi][2]]
                self.Pi_outlayers.append(Pi_outlayer)
                if Pi_outlayer is None:
                    nextoutsize = Pi_outsize
                else:
                    nextoutsize = Pi_outlayer.units
                self.Pi_nextoutsizes.append(nextoutsize)
                if pi is 0:
                    HIDDEN_SIZES = COMMON_HIDDEN_LAYER_SIZES + OUTPUT_HIDDEN_LAYER_SIZES
                else:
                    HIDDEN_SIZES = OUTPUT_HIDDEN_LAYER_SIZES
                self.Pi.append(PolicyModel(Pi_insize, Pi_outsize, HIDDEN_SIZES, trainable))

        #To calculate the probability of the output, and avoid duplicate code in learnerPAPPO
        def make_outprobs(self, pi, zchoices, logprobs, legal_ph):
            legal_row_len = legal_ph.shape[2]
            if pi < legal_row_len:
                depth = self.slicedims[pi][0]
                column = legal_ph[:, :, pi]
                nonneg = column >=0
                zeroed = tf.where(nonneg, column, tf.zeros_like(column))
                ohots = tf.one_hot(indices=zeroed, depth=depth, axis=-1, on_value=True, off_value=False, dtype=tf.bool)
                nohots = tf.zeros_like(ohots)
                nonneg3d = tf.tile(tf.expand_dims(nonneg, -1), [1, 1, depth])
                cohots = tf.where(nonneg3d, ohots, nohots)
                blegal = tf.reduce_any(cohots, axis=1)
                outprobs = tf.where(blegal, logprobs, logprobs - 1000.0)
                outprobs = tf.nn.log_softmax(outprobs)
            else:
                outprobs = logprobs

            if pi == 0:
                return outprobs
            else:
                dummy_outprobs = tf.pad(tf.constant([0.0]), [[0, self.slicedims[pi][0] -1]], constant_values=-1000.0)
                dummy_outprobs = tf.expand_dims(dummy_outprobs, 0)
                dummy_outprobs = tf.broadcast_to(dummy_outprobs, tf.shape(logprobs))
                triggerlist = self.slicedims[pi][1:]
                triggers = tf.constant(triggerlist, shape=[1, len(triggerlist)], dtype=tf.int32)
                check = tf.equal(zchoices, triggers)
                valid = tf.reduce_any(check, axis=1, keepdims=True)
                valid = tf.broadcast_to(valid, tf.shape(logprobs))

                return tf.where(valid, outprobs, dummy_outprobs)

        #To calculate the probability of pivalue, pivalue is used in PPO oldpi/current pi
        def make_pivalues(self, Pi_choices, Pi_outprobs):
            batchsize = tf.shape(Pi_outprobs[0])[0]
            itemnumbers = tf.expand_dims(tf.range(batchsize), -1)
            zerovecfloat = tf.zeros([batchsize], dtype=tf.float32)
            pivalues = zerovecfloat
            for pi in range(len(self.slicedims)):
                indices = tf.concat([itemnumbers, Pi_choices[pi]], axis=1)
                pivalues += tf.gather_nd(Pi_outprobs[pi], indices)
            pivalues = tf.exp(pivalues)
            return pivalues

        #To calculate the probability of the output, avoid duplicate code in learnerPAPPO
        def make_illegalsums(self, legal_ph, Pi_logprobs):
            legalpdf = tf.zeros_like(Pi_logprobs[0])
            illegalpdf = tf.zeros_like(Pi_logprobs[0])
            legal_row_len = legal_ph.shape[2]
            for pi in range(legal_row_len):
                depth = self.slicedims[pi][0]
                unlikely = tf.expand_dims(tf.fill([depth], -1000.), axis=0)
                unlikely = tf.broadcast_to(unlikely, tf.shape(Pi_logprobs[pi]))
                column = legal_ph[:, :, pi]
                nonneg = column >=0
                zeroed = tf.where(nonneg, column, tf.zeros_like(column))
                ohots = tf.one_hot(indices=zeroed, depth=depth, axis=-1, on_value=True, off_value=False, dtype=tf.bool)
                nohots = tf.zeros_like(ohots)
                nonneg3d = tf.tile(tf.expand_dims(nonneg, -1), [1, 1, depth])
                cohots = tf.where(nonneg3d, ohots, nohots)
                blegal = tf.reduce_any(cohots, axis=1)
                if pi == 0:
                    legalpdf = tf.where(blegal, Pi_logprobs[0], unlikely)
                    illegalpdf = tf.where(blegal, unlikely, Pi_logprobs[0])
                else:
                    triggerlist = self.slicedims[pi][1:]
                    triggers = tf.constant(triggerlist, shape=[len(triggerlist)], dtype=tf.int32)
                    trigonehots = tf.one_hot(triggers, depth=tf.shape(legalpdf)[0], axis=-1, on_value=True, off_value=False, dtype=tf.bool)
                    trigmask = tf.reduce_any(trigonehots, axis=0)
                    trigmask = tf.expand_dims(trigmask, axis=-1)
                    trigmask = tf.boardcast_to(trigmask, tf.shape(legalpdf))
                    plegal = tf.log(tf.reduce_sum(tf.exp(tf.where(blegal, Pi_logprobs[pi], unlikely)), axis=1))
                    plegal = tf.expand_dims(plegal, axis=-1)
                    plegal = tf.broadcast_to(plegal, tf.shape(legalpdf))
                    pillegal = tf.log(tf.reduce_sum(tf.exp(tf.where(blegal, unlikely,  Pi_logprobs[pi])), axis=1))
                    pillegal = tf.expand_dims(pillegal, axis=-1)
                    pillegal = tf.broadcast_to(pillegal, tf.shape(illegalpdf))
                    legalpdf = tf.where(trigmask, legalpdf+plegal, legalpdf)
                    illegalpdf = tf.where(trigmask, illegalpdf+pillegal, illegalpdf)
                return tf.reduce_sum(tf.exp(illegalpdf), axis=1) - tf.reduce_sum(tf.exp(legalpdf), axis=1)
        def call(self, inputs, legal_ph):
            newinput = self.embedlayer(inputs)
            Pi_insize =  newinput.shape[1]
            Pi_layers = []
            Pi_logprobs = []
            Pi_outprobs = []
            Pi_action_outputs = []
            Pi_nextout_inst = []
            Pi_choices = []

            for pi in range(len(self.slicedims)):
                if pi is 0:
                    Pi_input = newinput
                else:
                    Pi_input = Pi_layers[0][len(COMMON_HIDDEN_LAYER_SIZES)-1]

                for act in self.givens[pi]:
                    if act >= 0:
                        Pi_input = tf.concat([Pi_input, Pi_nextout_inst[act]], axis=1)
                Pi_layers.append(self.Pi[pi](Pi_input))
                Pi_logprobs.append(Pi_layers[-1][-1])

                if pi == 0:
                    outprobs = self.make_outprobs(pi, None, Pi_logprobs[-1], legal_ph)
                else:
                    outprobs = self.make_outprobs(pi, Pi_choices[0], Pi_logprobs[-1], legal_ph)
                Pi_outprobs.append(outprobs)

                choices = tf.random.categorical(outprobs, 1, dtype=tf.int32)
                Pi_choices.append(choices)

                onehots = tf.one_hot(choices, depth=outprobs.shape[1])
                onehots = tf.squeeze(onehots, axis=1)

                if self.actslices[pi][2] is 256:
                    Pi_action_outputs.append(self.Pi_outlayers[pi](onehots))
                else:
                    Pi_action_outputs.append(onehots)
                if self.Pi_outlayers[pi] is not None:
                    Pi_nextout_inst.append(self.Pi_outlayers[pi](onehots))
                else:
                    Pi_nextout_inst.append(onehots)

            actionvecs = tf.concat(Pi_action_outputs, axis=1)
            waitpivalues = tf.exp(Pi_logprobs[0][:,0])
            pivalues = self.make_pivalues(Pi_choices, Pi_outprobs)
            illegalsums = self.make_illegalsums(legal_ph, Pi_logprobs)
            return actionvecs, pivalues, waitpivalues, illegalsums, Pi_outprobs

    def __init__(self, modtype, modname, trainable=True):
        super().__init__(modtype, modname, trainable)
        
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as vs2:
            with tf.name_scope(vs2.original_name_scope):
                self.model = PolicyPAPPO.Model(self.modtype, trainable=trainable)        
    def predict_action(self, sess, statevec, legal_actions):
        self.action_vector, self.pivalues, self.waitpivalues, self.illegalsums, self.pi_outprobs = self.model(state, legal_actions)
        self.init_done()
        return self.action_vector, self.pivalues, self.waitpivalues, self.illegalsums

if __name__ == "__main__":

    import struct
    state_vector = np.array([0.5]*20 + [0.8]*63, dtype=np.float32)
    state_vector = np.stack((state_vector, state_vector, state_vector, state_vector))

    legal_list = np.array([[0, -1, -1],
                                [1, -1, -1],
                                [2, -1, -1],
                                [3, -1, -1],
                                [4, -1, -1],
                                [5, -1, -1],
                                [6, -1, -1],
                                [7, -1, -1],
                                [8, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1],
                                [-1, -1, -1]], dtype=np.int32)

    legal_list = np.stack((legal_list,  legal_list, legal_list, legal_list))
  
    class ModelType():
        def __init__(self):
            self.modid = 1
            self.statedim = None
            self.actdim  = None
            self.embedkernels  = None
            self.stateembed  = None
            self.actslices = None
            self.slicedims = None
            self.givens = None

    modtype = ModelType()
    modtype.statedim = 83
    modtype.actdim = 28
    modtype.embedkernels = [[9, 4], [9, 6], [2, 3], [3, 2], [4, 2], [12, 4]]   
    modtype.stateembed = [[0, 9, 0, 1, 0],
                            [20, 24, 4, 4, 10],
                            [24, 27, 3, 4, 10]]
    modtype.actslices = [[0, 12, 5],
                            [12, 16, 255],
                            [16, 25, 0],
                            [25, 26, 256],
                            [26, 27, 256],
                            [27, 28, 256]]
    modtype.slicedims = [[12], [4, 9], [9, 11], [61, 7, 8], [115, 1, 3, 4, 6], [77, 2, 3, 5, 6]]
    modtype.givens = [[-2], [-2], [-2], [0], [0], [0, 4]]


    class ModelName():
        def __init__(self):
            self.string = ''
    modname = ModelName()
    modname.string = "PAPPO"

    class LegalActions():
        actionlistlen = 20

    actionstr = struct.Struct(">lll")

    tensorboarddir = os.path.join(os.getcwd(), "tensorboard")
    #graph = tf.Graph()
    #graph_manager = graph.as_default()
    #graph_manager.__enter__()
    session = tf.Session()
    model = PolicyPAPPO(modtype, modname, trainable=True)

    state_vector = np.array([0.5]*20 + [0.8]*63, dtype=np.float32)
    state_vector = np.stack((state_vector, state_vector, state_vector, state_vector))


    action_vector, Pi_values, Wait_Pi_values, illegal_Pi_sum = model.predict_action(session, state_vector, legal_list)
    print("predicted action: ", action_vector)
    print("predicted Pi_values: ", Pi_values)
    print("predicted Wait_Pi_values: ", Wait_Pi_values)
    print("predicted illegal_Pi_sum: ", illegal_Pi_sum)
    

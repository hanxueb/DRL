"""
Implementation of PAPPO, based on papers:

Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space
https://arxiv.org/pdf/1903.01344.pdf

Hierarchical Approaches for Reinforcement Learning in Parameterized Action Space
https://arxiv.org/abs/1810.09656

Proximal Policy Optimization Algorithms
https://arxiv.org/abs/1707.06347


"""

import tensorflow as tf
tf.enable_eager_execution()


import numpy as np
import os
import sys
import time
import asyncio
import pickle

from pathlib import Path
import logging 
#logging.basicConfig()
log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

from learnerBase import learnerBase
from experience_buffer import ExperienceRecord, SegregatedExperienceBuffer

np.set_printoptions(linewidth=np.nan, suppress=True)

SHARED_LAYERS = 3
VALUE_HIDDEN_LAYER_SIZES = [40, 20]

class PappoSettings:
    ACTOR_LEARNING_RATE = 5e-4
    CRITIC_LEARNING_RATE = 5e-5

    DISCOUNT = 0.99

    UPDATE_TARGET_FREQ = 100
    UPDATE_TARGET_RATE = 0.2
    UPDATE_ACTOR_FREQ  =1

    SAVE_CHECKPOINT = 1000
    BATCH_SIZE = 32

    UPDATE_STEPS =1
    EP_MAX = 1000
    EP_LEN = 500

    SHARED_LAYERS = 0
    HIDDEN_LAYERS_SIZE = [200, 100]

    PPOEPSILON = 0.2
    C1 = 0.5
    C2 = 1.0
    C3 = 7.0
    C4 = 3.0

#Build the value network
class ValueModel(tf.keras.models.Model):
    def __init__(self, pimodel, hiddensize, trainable=True):
        super().__init__()
        self.pimodel = pimodel
        self.hiddensize = hiddensize
        self.hiddenlayers = []
        for size in self.hiddensize:
            self.hiddenlayers.append(tf.keras.layers.Dense(size, activation=tf.nn.selu, trainable=trainable))
        self.value = tf.keras.layers.Dense(1, activation=tf.nn.selu, trainable=trainable)

    def call(self, inputs):
        newinputs = self.pimodel.embedlayer(inputs)
        layers = [newinputs]
        for h in self.pimodel.Pi[0].hiddenlayers[0:SHARED_LAYERS]:
            layers.append(h(layers[-1]))
        for h in self.hiddenlayers:
            layers.append(h(layers[-1]))
        return self.value(layers[-1])

#Get pivalues, illegalsums, pi_entropy_loss from policy network created in learnerBase.py
class ActionPiModel(tf.keras.models.Model):
    def __init__(self, pimodel, trainable=True):
        super().__init__()
        self.model = pimodel

    def call(self, inputs, action_ph, legal_ph):
        newinputs = self.model.embedlayer(inputs)
        Pi_layers = []
        Pi_logprobs = []
        Pi_outprobs = []
        Pi_choices = []
        
        act_start = 0
        act_stop =0
        act_ph_slices = []
        Pi_nextout_inst = []

        for pi in range(len(self.model.slicedims)):
            act_start = self.model.actslices[pi][0]
            act_stop = self.model.actslices[pi][1]
            act_ph_slices.append(action_ph[:, act_start:act_stop])

            if self.model.Pi_outlayers[pi] is not None:
                if self.model.actslices[pi][2] < 255:
                    Pi_nextout_inst.append(self.model.Pi_outlayers[pi](action_ph[:, act_start:act_stop]))
                else:
                    Pi_nextout_inst.append(action_ph[:, act_start:act_stop])

        for pi in range(len(self.model.slicedims)):
            if pi is 0:
                 Pi_input = newinputs
            else:
                Pi_input = Pi_layers[0][SHARED_LAYERS -1]
            for act in self.model.givens[pi]:
                if act >= 0:
                    Pi_input = tf.concat([Pi_input, Pi_nextout_inst[act]], axis=1)
            Pi_layers.append(self.model.Pi[pi](Pi_input))
            Pi_logprobs.append(Pi_layers[-1][-1])

            if pi == 0:
                outprobs = self.model.make_outprobs(pi, None, Pi_logprobs[-1], legal_ph)
            else:
                outprobs = self.model.make_outprobs(pi, Pi_choices[0], Pi_logprobs[-1], legal_ph)
            Pi_outprobs.append(outprobs)
            
            if self.model.actslices[pi][2] is 256:
                choices = tf.cast(act_ph_slices[pi]*self.model.slicedims[pi][0], tf.int32)
            else:
                #no back propogation for argmax
                choices = tf.argmax(act_ph_slices[pi], axis=1, output_type=tf.int32)
                choices = tf.expand_dims(choices, -1)
            Pi_choices.append(choices)

        pivalues = self.model.make_pivalues(Pi_choices, Pi_outprobs)
        illegalsums = self.model.make_illegalsums(legal_ph, Pi_logprobs)
        pi_entropy_loss = tf.zeros([tf.shape(inputs)[0]], dtype=tf.float32)
        for pi in range(3):
            pi_entropy_loss = pi_entropy_loss + tf.reduce_sum(Pi_outprobs[pi]*tf.exp(Pi_outprobs[pi]), axis=1)
        return pivalues, illegalsums, pi_entropy_loss

class PAPPONetwork(learnerBase):
    algorithm = "PAPPO"
    
    def __init__(self, modtype, modname, trainable=True):
        super().__init__(modtype, modname, trainable)      
        #with self.graph.as_default():
        self.kerneldims = modtype.embedkernels
        self.stateembed = modtype.stateembed
        self.actslices = modtype.actslices
        self.slicedims = modtype.slicedims
            
        self.scopestr = self.modname.string + "_" + str(self.modtype.modid) + "_learner"
        with tf.variable_scope(self.scopestr) as self.scope:
            self.value_network = ValueModel(self.actor.model, VALUE_HIDDEN_LAYER_SIZES)
            self.actionpidmodel = ActionPiModel(self.actor.model, trainable=True)
            
        self.train_optimizer = tf.train.AdamOptimizer(PappoSettings.CRITIC_LEARNING_RATE)

        self.summary_writer = tf.contrib.summary.create_file_writer(Path("./tensorboard"), flush_millis=10000)
        #self.init_done()
        log.debug("PAPPO is created")

    def compute_loss(self, batch):
        
        state_batch = np.array([exp_record.state for exp_record in batch])
        action_batch = np.array([exp_record.action for exp_record in batch])
        mu_batch = np.array([exp_record.pi for exp_record in batch])
        next_state_batch = np.array([exp_record.nextstate for exp_record in batch])
        reward_batch = np.array([exp_record.reward for exp_record in batch])
        duration_batch = np.array([exp_record.duration for exp_record in batch])
        eps_done_batch = np.array([exp_record.doneflag for exp_record in batch])
        legal_acts_batch = np.array([exp_record.legal_acts for exp_record in batch])

        gammas = (1.0 - eps_done_batch)*np.power(PappoSettings.DISCOUNT, np.maximum(duration_batch, 0.01))
        unitsquare =  np.full((len(duration_batch), len(duration_batch)), 1.0, dtype=np.float32)
        udunit = np.triu(unitsquare)
        gammasquare = np.copy(udunit)
        for b in range(len(duration_batch)-1):
            mul = np.copy(udunit)
            mul[:b+1, b+1:] = gammas[b]
            gammasquare = gammasquare*mul

        self.values_current_states = self.value_network(state_batch)
        self.values_next_states = self.value_network(next_state_batch)

        self.tderrors = reward_batch + gammas*self.values_next_states - self.values_current_states
        self.tdsquare = tf.tile(self.tderrors, [1, len(batch)])
        self.tdsquare = tf.transpose(self.tdsquare)

        #needs further work, add lambda in formula 11 in PPO paper
        self.advantages = tf.reduce_sum(self.tdsquare*gammasquare, axis=1)
        value_loss = tf.squeeze(tf.square(self.tderrors), axis=1)
        piold = mu_batch
        piold = tf.clip_by_value(piold, 1.e-6, 1.)
        self.actor_pivalues, self.actor_illeaglpisums, entropy_loss = \
            self.actionpidmodel(state_batch, action_batch, legal_acts_batch)
        ratio = self.actor_pivalues / piold
        clipped_ratio = tf.clip_by_value(ratio, 1. - PappoSettings.PPOEPSILON, 1. + PappoSettings.PPOEPSILON)
        surrogate = ratio*self.advantages
        clipsurrogate = clipped_ratio*self.advantages
        policy_loss = -tf.minimum(surrogate, clipsurrogate)
        illegal_loss = self.actor_illeaglpisums
        total_loss = tf.reduce_mean(PappoSettings.C1 * value_loss + PappoSettings.C2 * policy_loss + \
                                    PappoSettings.C3 * illegal_loss + PappoSettings.C4 * entropy_loss)
        self.policy_variables = self.actor.model.variables
        self.learner_variables = self.value_network.variables + self.actionpidmodel.variables
        self.all_parameters = self.policy_variables + self.learner_variables
        
        return total_loss, value_loss, policy_loss, illegal_loss, entropy_loss

    async def run(self):
        #await self.restore()
        global_step = tf.train.get_or_create_global_step()
        checkpoint_directory = Path("./checkpoint")
        checkpoint_prefix = Path(checkpoint_directory, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.train_optimizer,
                                   #model=model,
                                   optimizer_step=global_step)
        try:
            ckptfile = self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
            if ckptfile is None:
                log.debug("Could not restore checkpoint")
            else:
                log.debug("PPO checkpoint is restored")
        except:
            log.debug("Exception in checkpoint restore")

        self.total_eps = 0
        self.iteration_number = 0
        start_time = time.time()
        while(True):
            batch =  self.exp_buff.sample(duration=8.0)
            if(len(batch) == 0):
               log.debug("No eligible UEs in PAPPO train(out of %d UEs", len(self.exp_buff))
               await asyncio.sleep(3)
               start_time = time.time()
               continue
            with tf.GradientTape() as tape:
                self.total_loss, self.value_loss, self.policy_loss, self.illegal_loss, self.entropy_loss = self.compute_loss(batch)
                grads = tape.gradient(self.total_loss, self.learner_variables)
                self.train_optimizer.apply_gradients(zip(grads,  self.learner_variables),
                            global_step=tf.train.get_or_create_global_step())
           
            self.total_eps +=1
            if self.total_eps % 100 == 0:
                log.debug("PPO value loss: %f, policy_loss: %f, illegal loss: %f, entropy_loss: %f, in eps: %d", \
                    self.value_loss, self.policy_loss, self.illegal_loss, self.entropy_loss, self.total_eps)
                with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    self.value_loss_summary = tf.contrib.summary.scalar("PAPPO_value_loss", tf.reduce_mean(self.value_loss))
                    self.policy_loss_summary = tf.contrib.summary.scalar("PAPPO_policy_loss", tf.reduce_mean(self.policy_loss))
                    self.illegal_loss_summary = tf.contrib.summary.scalar("PAPPO_illegal_loss", tf.reduce_mean(self.illegal_loss))
                    self.entropy_loss_summary = tf.contrib.summary.scalar("PAPPO_entropy_loss", tf.reduce_mean(self.entropy_loss))

            if self.total_eps % PappoSettings.SAVE_CHECKPOINT ==0:
                self.iteration_number +=1
                log.debug("PPO iteration %d takes %f seconds", self.iteration_number, time.time() - start_time)
                save_path = self.checkpoint.save(file_prefix=checkpoint_prefix)
                log.debug("PPO checkpoint is saved to: %s", save_path)
                start_time = time.time()

if __name__ == "__main__":

    logging.basicConfig()
    log.setLevel(logging.DEBUG)

    import struct
    from policyPAPPO import PolicyPAPPO

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
            self.algotype = None

    modtype = ModelType()
    
    #state vector length
    modtype.statedim = 83

    #action vector length
    modtype.actdim = 28
    #embed kernel [inuptsize, outputsize]
    modtype.embedkernels = [[9, 4], [9, 6], [2, 3], [3, 2], [4, 2], [12, 4]]   
    
    #embedding [start, stop, kernel index, repeat number, repeat interval]
    modtype.stateembed = [[0, 9, 0, 1, 0],
                            [20, 24, 4, 4, 10],
                            [24, 27, 3, 4, 10]]
    # action slice [start, stop, embedding index or 255 means no embedding or 256 means rescale is needed]
    modtype.actslices = [[0, 12, 5],
                            [12, 16, 255],
                            [16, 25, 0],
                            [25, 26, 256],
                            [26, 27, 256],
                            [27, 28, 256]]
    #for each network, [outputsize,  triggerlist means which action(not defined here) will trigger network ]
    #The first network output is the top level actions, no tirggerlist
    #when the first network predicted action 7 or 8, the prediction from network [61,7,8] is needed to compose 
    # 1 complete action for agent
    modtype.slicedims = [[12], [4, 9], [9, 11], [61, 7, 8], [115, 1, 3, 4, 6], [77, 2, 3, 5, 6]]

    #how to chain the network, -2 means no chain, 0 means get the first network's output
    modtype.givens = [[-2], [-2], [-2], [0], [0], [0, 4]]
    modtype.algotype = "PAPPO"


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
    #session = tf.Session()
    state = np.array([0.5]*20 + [0.8]*63)
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
                                    [-1, -1, -1]])

    model = PAPPONetwork(modtype, modname, trainable=True)

    #simulate some input data
    for i in range(100):
        state = np.array([0.5]*20 + [0.8]*63, dtype=np.float32)
        action = np.array([0.6]*28,  dtype=np.float32)
        pi = np.array([1.0], dtype=np.float32)
        nextstate = np.array([0.5]*20 + [0.8]*63, dtype=np.float32)
        reward =np.array([1.0], dtype=np.float32)
        duration = np.array([5.0], dtype=np.float32)
        episodedoneflag = 0
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
                                    [-1, -1, -1]])
        exp = ExperienceRecord(state, action, pi, nextstate, reward, duration, episodedoneflag, legal_list)
        model.exp_buff.add(i, exp)

    #model.sess = tf.Session(graph=model.graph)
    model.loop = asyncio.get_event_loop()
    #model.tb_writer = tf.summary.FileWriter(tensorboarddir, model.graph)
    model.loop.create_task(model.run())
    model.loop.run_forever()

    
    


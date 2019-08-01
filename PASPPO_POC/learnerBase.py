"""
Base class for different algorithms

"""

import tensorflow as tf
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


import policy
from experience_buffer import ExperienceRecord, SegregatedExperienceBuffer

class Factory(object):
    classmap = {}
    @staticmethod
    def create(modtype, modname, trainable=True):
        return Factory.classmap[modtype.algotype](modtype, modname, trainable)

    @staticmethod
    def register(cls):
        Factory.classmap[cls.algorithm] = cls

class Register(type):
    def __init__(cls, name, bases, dct):
        type.__init__(cls, name, bases, dct)
        Factory.register(cls)

class learnerBase(object, metaclass=Register):
    algorithm =None

    async def run_sess(self, *args, **kwargs):
        return await self.loop.run_in_executor(None, lambda: self.sess.run(*args, **kwargs))

    def __init__(self, modtype, modname, trainable=True):
        log.debug("Learnerbase: modtype: %s, modname: %s", str(modtype), str(modname))
        self.modtype = modtype
        self.modname = modname
        self.statedim = modtype.statedim
        self.actdim = modtype.actdim
        self.algotype = modtype.algotype

        self.loop = None
        self.exp_buff = None
        self.weightspub = None
        self.tb_writer = None
        self.sess = None
        self.run_task = None
        self.graph = tf.Graph()

        self.iteration_number = 0
        self.total_eps = 0
        allparamdir = os.path.join(os.getcwd(), "all_params_simPAPPO")
        if not os.path.exists(allparamdir):
            os.mkdir(allparamdir)
        self.all_params_file = os.path.join(allparamdir, "params_" + str(self.statedim) + "_" + str(self.actdim) + ".ckpt")
        expdir = os.path.join(os.getcwd(), "exp_params_simPAPPO")
        if not os.path.exists(expdir):
            os.mkdir(expdir)
        self.exp_file = os.path.join(expdir, "exp_" + str(self.modtype.modid) + "_" + str(self.modtype.algotype) + "_" + str(self.statedim) + "_" + str(self.actdim) + ".pkl")
        self.mtype_file = os.path.join(expdir, "mtype_" + str(self.modtype.modid) + "_" + str(self.modtype.algotype) + "_" + str(self.statedim) + "_" + str(self.actdim) + ".pkl")

        log.debug("Calling policy.Factory.create with %s", self.modname.string)
        #with self.graph.as_default():
        self.actor = policy.Factory.create(self.modtype, self.modname, trainable)
            #self.input = self.actor.input

        try:
            with open(self.exp_file, 'rb') as f:
                self.exp_buff = pickle.load(f)
                log.debug("Loaded experience buffer of length %d", len(self.exp_buff))
        except:
            log.debug("Could not load experience from ", self.exp_file)
            self.exp_buff = SegregatedExperienceBuffer(300)
    def init_done(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(self.all_parameters)
            self.init_globals = tf.global_variables_initializer()
            self.graph.finalize()

    async def save(self):
        save_path = await self.actor.save_weights(self.loop, self.sess)
        if self.weightspub is not None:
            filenametlv = ParamFanme(None, save_path)
            self.weightspub.publish_weights([self.modtype, filenametlv])
        save_path =  await self.loop.run_in_executor(None, lambda: self.saver.save(self.sess, self.all_params_file))
        log.debug("Successfully saved %s", str(self.all_params_file))

    async def restore(self):
        try:
            await self.loop.run_in_executor(None, lambda: self.saver.save(self.sess, self.all_params_file))
            log.debug("Successfully restored %s", str(self.modtype))
            await self.save()
        except:
             log.debug("Could not restore %s", str(self.modtype))

    def start(self, loop, weightspub, tb_writer):
        if self.run_task is not None and not self.run_task.done():
            return
        if self.run_task is not None:
            try:
               self.run_task.result()
            except:
                log.exception("In %s.run()", self.modname.string)
        log.debug("%s: %s (%d) start", self.modtype.algotype.name, self.modname.string, self.modtype.modid)
        self.sess = tf.Session(graph=self.graph)
        self.loop = loop
        self.weightspub = weightspub
        self.tb_writer = tb_writer
        self.run_task = self.loop.create_task(self.run())

    def dump_experience():
        try:
            with open(self.exp_file, 'wb+') as f:
                pickle.dump(self.exp_buff, f)
                log.debug("Saved experience length %d to %s", len(self.exp_buff), self.exp_file)
        except:
            log.debug("Could not save experience to %s", self.exp_file)

    def stop(self):
        if Settings.RUN_LEARNER_ONLY is False and self.exp_buff is not None:
            self.dump_experience()
        else:
            log.debug("No new experience to dump")
        if self.run_task is not None and not self.run_task.done():
            self.run_task.cancel()
        self.run_task = None
        self.iteration_number = 0
        if self.sess is not None:
            self.sess.close()
            self.sess = None

#import learnerPASPPO


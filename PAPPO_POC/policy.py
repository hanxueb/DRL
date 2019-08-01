"""
Base class for different policy networks

"""

import tensorflow as tf
import numpy as np
import logging
import asyncio
import os

log = logging.getLogger(__name__)


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

class Policy(object, metaclass=Register):
    algorithm = None

    def __init__(self, modtype, modname, trainable=True):

        self.modtype = modtype
        self.modname = modname
        self.modid = modtype.modid
        self.statedim = modtype.statedim
        self.actdim = modtype.actdim

        actorparamdir = os.path.join(os.getcwd(), "actor_params_simPAPPO")
        if not os.path.exists(actorparamdir):
            os.mkdir(actorparamdir)
        self.actor_params_file = os.path.join(actorparamdir, "params_" + str(self.statedim) + "_" + str(self.actdim) + ".ckpt")
        self.scopestr = self.modname.string + "_" + str(self.modid) + "_policy"
        with tf.variable_scope(self.scopestr) as self.scope:
            pass
        #    self.input =  state_vector

    def init_done(self):
        self.actor_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope="^" + self.scopestr)
        self.saver = tf.train.Saver(self.actor_parameters)
    async def load_weights(self, loop, sess, fname=None):
        global_step = tf.train.get_or_create_global_step()
        checkpoint_directory = Path("./checkpoint")
        checkpoint_prefix = Path(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint()
        try:
            ckptfile = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
            if ckptfile is None:
                log.debug("Could not restore policy checkpoint")
                return False
            else:
                log.debug("PPO policy checkpoint is restored")
                return True
        except:
            log.debug("Exception in policy checkpoint restore")
            return False

    async def save_weights(self, loop, sess, fname=None):
        if fname is None:
            fname = self.actor_params_file
        save_path =  await loop.run_in_executor(None, 
                                               lambda: self.saver.save(sess, fname))
        return save_path

import policyPAPPO



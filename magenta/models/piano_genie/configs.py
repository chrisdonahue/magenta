# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hyperparameter configurations for Piano Genie."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BasePianoGenieConfig(object):
  """Base class for model configurations."""

  def __init__(self):
    # Data parameters
    self.data_max_discrete_times = 32
    self.data_max_discrete_velocities = 16
    self.data_randomize_chord_order = False

    # RNN parameters (encoder and decoder)
    self.rnn_celltype = "lstm"
    self.rnn_nlayers = 2
    self.rnn_nunits = 128

    # Encoder parameters
    self.enc_rnn_bidirectional = True
    self.enc_pitch_scalar = False
    self.enc_aux_feats = []

    # Decoder parameters
    self.dec_autoregressive = False
    self.dec_aux_feats = []
    self.dec_pred_velocity = False

    # Unconstrained "discretization" parameters
    # Passes sequence of continuous embeddings directly to decoder (which we
    # will discretize during post-processing i.e. with K-means)
    self.stp_emb_unconstrained = False
    self.stp_emb_unconstrained_embedding_dim = 64

    # Integer quant parameters
    self.stp_emb_iq = False
    self.stp_emb_iq_nbins = 8
    self.stp_emb_iq_contour_dy_scalar = False
    self.stp_emb_iq_contour_margin = 0.
    self.stp_emb_iq_contour_exp = 2
    self.stp_emb_iq_contour_comp = "product"
    self.stp_emb_iq_deviate_exp = 2

    # Unconstrained parameters... just like VAE but passed directly (no prior)
    self.seq_emb_unconstrained = False
    self.seq_emb_unconstrained_embedding_dim = 64

    # VAE parameters. Last hidden state of RNN will be projected to a summary
    # vector which will be passed to decoder with Gaussian re-parameterization.
    self.seq_emb_vae = False
    self.seq_emb_vae_embedding_dim = 64

    # (lo)w-(r)ate latents... one per every N steps of input
    self.lor_emb_n = 16
    self.lor_emb_unconstrained = False
    self.lor_emb_unconstrained_embedding_dim = 8

    # Training parameters
    self.train_batch_size = 32
    self.train_seq_len = 128
    self.train_seq_len_min = 1
    self.train_randomize_seq_len = False
    self.train_augment_stretch_bounds = (0.95, 1.05)
    self.train_augment_transpose_bounds = (-6, 6)
    self.train_loss_iq_range_scalar = 1.
    self.train_loss_iq_contour_scalar = 1.
    self.train_loss_iq_deviate_scalar = 0.
    self.train_loss_vae_kl_scalar = 1.
    self.train_lr = 3e-4

    # Eval parameters
    self.eval_batch_size = 32
    self.eval_seq_len = 128


class StpFree(BasePianoGenieConfig):

  def __init__(self):
    super(StpFree, self).__init__()

    self.stp_emb_unconstrained = True


class StpIq(BasePianoGenieConfig):

  def __init__(self):
    super(StpIq, self).__init__()

    self.stp_emb_iq = True


class SeqFree(BasePianoGenieConfig):

  def __init__(self):
    super(SeqFree, self).__init__()

    self.seq_emb_unconstrained = True


class SeqVae(BasePianoGenieConfig):

  def __init__(self):
    super(SeqVae, self).__init__()

    self.seq_emb_vae = True


class LorFree(BasePianoGenieConfig):

  def __init__(self):
    super(LorFree, self).__init__()

    self.lor_emb_unconstrained = True


class Auto(BasePianoGenieConfig):

  def __init__(self):
    super(Auto, self).__init__()

    self.dec_autoregressive = True


class StpIqAuto(BasePianoGenieConfig):

  def __init__(self):
    super(StpIqAuto, self).__init__()

    self.stp_emb_iq = True
    self.dec_autoregressive = True


class SeqVaeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(SeqVaeAuto, self).__init__()

    self.seq_emb_vae = True
    self.dec_autoregressive = True


class LorFreeAuto(BasePianoGenieConfig):

  def __init__(self):
    super(LorFreeAuto, self).__init__()

    self.lor_emb_unconstrained = True
    self.dec_autoregressive = True


_named_configs = {
    "stp_free": StpFree(),
    "stp_iq": StpIq(),
    "seq_free": SeqFree(),
    "seq_vae": SeqVae(),
    "lor_free": LorFree(),
    "auto_no_enc": Auto(),
    "stp_iq_auto": StpIqAuto(),
    "seq_vae_auto": SeqVaeAuto(),
    "lor_free_auto": LorFreeAuto(),
}


def get_named_config(name, overrides=None):
  """Instantiates a config object by name.

  Args:
    name: Config name (see _named_configs)
    overrides: Comma-separated list of overrides e.g. "a=1,b=2"

  Returns:
    cfg: The config object
    summary: Text summary of all params in config
  """
  cfg = _named_configs[name]

  if overrides is not None and overrides.strip():
    overrides = [p.split("=") for p in overrides.split(",")]
    for key, val in overrides:
      val_type = type(getattr(cfg, key))
      if val_type == bool:
        setattr(cfg, key, val in ["True", "true", "t", "1"])
      elif val_type == list:
        setattr(cfg, key, val.split(";"))
      else:
        setattr(cfg, key, val_type(val))

  summary = "\n".join([
      "{},{}".format(k, v)
      for k, v in sorted(vars(cfg).items(), key=lambda x: x[0])
  ])

  return cfg, summary

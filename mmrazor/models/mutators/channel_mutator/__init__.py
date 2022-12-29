# Copyright (c) OpenMMLab. All rights reserved.
from .channel_mutator import ChannelMutator
from .dcff_channel_mutator import DCFFChannelMutator
from .group_fisher_channel_mutator import GroupFisherChannelMutator
from .one_shot_channel_mutator import OneShotChannelMutator
from .slimmable_channel_mutator import SlimmableChannelMutator

__all__ = [
    'SlimmableChannelMutator', 'ChannelMutator', 'OneShotChannelMutator',
    'DCFFChannelMutator', 'GroupFisherChannelMutator'
]

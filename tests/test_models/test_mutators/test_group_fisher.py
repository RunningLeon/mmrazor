# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.mutators import GroupFisherChannelMutator
from ...data.models import SingleLineModel


class TestGroupFisher(unittest.TestCase):

    def test_group_fisher_mutator(self):
        model = SingleLineModel()
        mutator = GroupFisherChannelMutator()
        mutator.prepare_from_supernet(model)

        mutator.start_pruning()
        for _ in range(5):
            x = torch.rand([1, 3, 224, 224])
            model(x)
            mutator.try_prune()
        mutator.end_pruning()
        print(model)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule primitive base."""
from __future__ import annotations
from abc import abstractmethod

PRIMITIVES = {}


def register_primitive():
    """Register a primitive to the schedule."""

    def dectorator(cls):

        if cls.name() in PRIMITIVES:
            raise ValueError(f"Primitive {cls.name()} already registered")
        if not issubclass(cls, Primitive):
            raise ValueError(f"Class {cls} is not a subclass of Primitive")
        PRIMITIVES[cls.name()] = cls
        return cls

    return dectorator


class Primitive:
    """A base class of schedule primitives."""

    @staticmethod
    @abstractmethod
    def name():
        """The name of the primitive."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def apply(sch, *args, **kwargs):
        """Apply the primitive to the schedule."""
        raise NotImplementedError

    @staticmethod
    def init_metadata():
        """(Optional) Initialize the metadata of the primitive."""
        return None

"""Parent selection: the aspect of the EA this research question varies.

Both selectors take the list of current parents (evaluated `Individual`s whose
tags hold the five per-target distances) and return ONE chosen parent.
"""

# Standard library
import random
from collections.abc import Callable, Sequence

# Local libraries (ARIEL)
from ariel.ec import Individual

# Local scripts
from common import N_TARGETS

type Selector = Callable[[list[Individual]], Individual]


def tournament(parents: list[Individual], k: int = 2) -> Individual:
    """Draw k distinct bodies at random; the lowest mean + std wins."""
    return min(random.sample(parents, k), key=lambda ind: ind.fitness)


def lexicase(parents: list[Individual], order: Sequence[int] | None = None) -> Individual:
    """Filter the parents one target at a time, in a fresh random order.

    On each target only the bodies with the lowest distance survive; stop when
    one is left, or pick at random among the survivors once all targets are
    used. `order` fixes the target order (only for testing).
    """
    candidates = list(parents)
    cases = order if order is not None else random.sample(range(N_TARGETS), N_TARGETS)
    for case in cases:
        best = min(ind.tags["dists"][case] for ind in candidates)
        candidates = [ind for ind in candidates if ind.tags["dists"][case] == best]
        if len(candidates) == 1:
            break
    return random.choice(candidates)


def make_selector(name: str, k: int = 2) -> Selector:
    """Return the selector for a condition name ("tournament" or "lexicase")."""
    match name:
        case "tournament":
            return lambda parents: tournament(parents, k)
        case "lexicase":
            return lexicase
        case _:
            msg = f"unknown selection scheme: {name!r}"
            raise ValueError(msg)

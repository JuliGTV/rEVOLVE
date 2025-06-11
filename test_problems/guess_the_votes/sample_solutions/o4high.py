from itertools import product
from typing import Dict, Set

def guess_the_votes(shares: Dict[str, int], votes: Dict[str, int]) -> Dict[str, Set[str]]:
    owners = list(shares.keys())
    options = list(votes.keys())

    # 1) Enumerate all possible assignments owner → option
    valid_assignments = []
    for assignment in product(options, repeat=len(owners)):
        # build a map owner->option
        assign_map = dict(zip(owners, assignment))
        # check whether the weighted sums match
        tally = {opt: 0 for opt in options}
        for owner, opt in assign_map.items():
            tally[opt] += shares[owner]
        if tally == votes:
            valid_assignments.append(assign_map)

    # 2) For each owner, see if they vote the same in all valid assignments
    result: Dict[str, Set[str]] = {opt: set() for opt in options}
    if not valid_assignments:
        return result

    for owner in owners:
        # take their vote in the first valid assignment
        first_vote = valid_assignments[0][owner]
        # check consistency across all
        if all(assign[owner] == first_vote for assign in valid_assignments):
            result[first_vote].add(owner)

    return result



from typing import Dict, Set
from itertools import permutations

def guess_the_votes(shares: Dict[str, int], votes: Dict[str, int]) -> Dict[str, Set[str]]:
    from itertools import permutations
    
    people = list(shares.keys())
    share_values = list(shares.values())
    total_votes = sum(votes.values())

    # Start with an empty mapping from vote options to sets of people
    result = {option: set() for option in votes}

    # Get all possible assignments (permutations) of people to vote options
    all_vote_options = list(votes.keys())
    n = len(people)

    valid_assignments = []

    # All possible assignments of people to vote options
    def generate_assignments(people, options):
        if not people:
            return [[]]
        results = []
        for option in options:
            for sub_assignment in generate_assignments(people[1:], options):
                results.append([(people[0], option)] + sub_assignment)
        return results

    all_assignments = generate_assignments(people, all_vote_options)

    # Check which assignments match the total vote share per option
    for assignment in all_assignments:
        vote_totals = {option: 0 for option in all_vote_options}
        for person, option in assignment:
            vote_totals[option] += shares[person]
        if vote_totals == votes:
            valid_assignments.append(assignment)

    # For each person, check if they always voted the same option in all valid assignments
    for person in people:
        voted_options = set()
        for assignment in valid_assignments:
            for p, option in assignment:
                if p == person:
                    voted_options.add(option)
        if len(voted_options) == 1:
            # The person always voted the same way in all valid assignments
            result[voted_options.pop()].add(person)

    return result

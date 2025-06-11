from example_problems.guess_the_votes.sample_solutions import o4high_golf, o4high, gpt4o, gpt4o_golf
from typing import Callable
import json


def test_guess_the_votes(gtv: Callable[[dict[str, int], dict[str, int]], dict[str, set[str]]]) -> bool:
    # Load test cases from JSON file
    with open('test_problems/guess_the_votes/test.json', 'r') as f:
        test_cases = json.load(f)
    
    for test_case in test_cases:
        shares = test_case['shares']
        votes = test_case['votes']
        # Convert expected lists to sets for comparison
        expected = {k: set(v) for k, v in test_case['expected'].items()}
        
        result = gtv(shares, votes)
        if result != expected:
            print(f"Test case {shares, votes} failed")
            print(f"Expected: {expected}")
            print(f"Got: {result}")
            return False
    return True

if __name__ == "__main__":
    print(test_guess_the_votes(o4high.guess_the_votes))
    print(test_guess_the_votes(o4high_golf.guess_the_votes))
    print(test_guess_the_votes(gpt4o.guess_the_votes))
    print(test_guess_the_votes(gpt4o_golf.guess_the_votes))
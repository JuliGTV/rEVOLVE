# Guess the votes problem (code golf)

I have found the following problem on codewars (https://www.codewars.com/kata/681500c89d879c62b24a592d)

We will be giving the problem to the model as a code golf problem where the goal is to give the shortest possible valid solution (in terms of characters used)

## problem

This problem is inspired by a real-world scenario involving shared ownership and voting.

In certain entities — such as apartment buildings or cooperatives — owners hold unequal shares, and votes are weighted according to ownership share. Although voting is supposed to be anonymous, it's often possible to deduce who voted how using public information about share distributions and final vote results.

Task
Implement a function guess_the_votes(shares: Dict[str, int], votes: Dict[str, int]) -> Dict[str, Set[str]].

You must try uncover as many people's votes as possible. For each person which you can be sure voted for a particular outcome, place them in the relevant set in the output. If you cannot be completely certain of a person's vote, then leave them out of the output.

Input
The input will consist of two dictionaries (dict in python):

shares mapping names to share sizes

votes mapping possible vote options to number of shares voting for this options

The input will always be valid. Sum of shares will always be the same as sum of all votes.

Number of different voting options will be 2 or 3. The number of shares will not be higher than 8.

Output
Return a dict mapping each vote option to a set of share owner names which you can be sure voted for this particular outcome.

All vote options must be included in the result dictionary.

Examples
In the first example all votes can be deanonymized.
```
shares = { "Perfectionists": 10, "Impatient": 20, "Quick-tempered": 20, "Scrooge": 50 }
votes = { "Better": 10, "Faster": 40, "Cheaper": 50 }
guess_the_votes(shares, votes) == {'Better': {'Perfectionists'}, 'Faster': {'Impatient', 'Quick-tempered'}, 'Cheaper': {'Scrooge'}}
```
In the example none of the votes can be deanonymized.
```
shares = {"A": 10, "B": 20, "C": 20, "D": 50 }
votes = {"Yes": 50, "No": 50 }
guess_the_votes(shares, votes) = {'Yes': set(), 'No': set()}
```
In the following example only some votes can be deanonymized.

``` 
shares = { "Dad": 7, "Kid1": 2, "Kid2": 2 }
votes = { "Ice Cream": 9, "Chocolate": 2, "Homework": 0 }
guess_the_votes(shares, votes) == { 'Ice Cream': {'Dad'}, 'Chocolate': set(), 'Homework': set() }
```



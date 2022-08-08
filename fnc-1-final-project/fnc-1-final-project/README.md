# Custom FNC implementation

Information about the fake news challenge can be found on [FakeChallenge.org](http://fakenewschallenge.org).


### Hold-out split (for development)


|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    431    |    14     |    284    |    33     |
-------------------------------------------------------------
| disagree  |    53     |    43     |    61     |     5     |
-------------------------------------------------------------
|  discuss  |    135    |    30     |   1584    |    51     |
-------------------------------------------------------------
| unrelated |    12     |     5     |    51     |   6830    |
-------------------------------------------------------------
Score: 3909.75 out of 4448.5	(87.8891761267843%)

Best score received in leaderboard with LongT5 summarization + Distilbert with 10-fold cross validation classifier running on 200 epochs. 
Score: 9493.5 out of 11651.25     (81.48%)

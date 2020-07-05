# Dimensions
#   Card value: 1 - 7462
#     linear? log? - n bins
#   Own pot investment - m bins
#   Pot investment of second biggest player - o bins
#   Pot investment of third biggest player - p bins
#   Pot investment of fourth biggest player - q bins
#   Pot investment of fifth biggest player - r bins
#   Pot investment of last player - s bins
#   Seating position - 6
#   Round - 4

# m = o = p = q = r = s
#   = 4
#     ==> total = 98'304 * n bins
#   = 6
#     ==> total = 1'119'744 * n bins

# Different approach: either players are in the running, or they're not
# instead of comparing each player pot investment, compare total pot from players who are out,
# and how many players are still in
#   Own pot investment - m bins
#   Total pot size - u bins
#   players still in - 5
#     m = 6, u = 8
#       ==> total = 5760 * n bins (more or less, minus invalid states)

# Card value: Use percentile hand rank of that number of cards
#   different distribution per round type
#   for preflop: use percentile preflop quality table from the other monte carlo
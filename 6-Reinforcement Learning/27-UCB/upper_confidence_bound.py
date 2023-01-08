# Upper Confidence Bound

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB using via scratch not via function
import math
N = dataset.shape[0] # = 10000 # number of different - different rounds.
d = dataset.shape[1] # = 10  # number of different - different versions of ad.
ads_selected = [] # Empty Vector
numbers_of_selections = [0] * d # Vector of size d # Vector of zero symbolizies that at first round no ad has been selected. 
sum_of_rewards = [0] * d # Vector of size d # Vector of zero symbolizies that at first round no ad has been selected. 
total_rewards = 0
    
for n in range(0, N):   # this for loop is used to select each round.
    ad = 0
    max_upper_bound = 0 # diffrent for each round.
    for i in range(0, d): # this for loop is used to select each version of ad for specifiec round.
        # initial 10 first round we don't have any data to observe so initial all get selected and they can't be taken reward.
        if numbers_of_selections[i] > 0:
            average_reward = sum_of_rewards[i] / numbers_of_selections[i] # taking the ith number of element 
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # larger value of this one is only avoid rest of 9 round to select the max upper bound.
                                # for second round 2nd ad is get selected.
                                # such as for 10 round respectively 10 ad get selected.
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i # track the max upper bound
    ads_selected.append(ad) # all the different ads get selected in each round.                    
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 # update the number of selection of particular ad in each round.
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_rewards = total_rewards + reward
    
# Visualing the results
plt.hist(ads_selected)    
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times the ad was selected')
plt.show()    
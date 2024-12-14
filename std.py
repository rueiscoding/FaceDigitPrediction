import math

def standardDeviation (data):
    mean = sum(data) / len(data)
    squared_diff_sum = 0
    for x in data:
        squared_diff = (x - mean) ** 2
        squared_diff_sum = squared_diff_sum + squared_diff
        variance = squared_diff_sum / (len(data) - 1)
        standard_deviation = math.sqrt (variance)
        return standard_deviation
    

# at one percent
outcome_testingaccuracy = 0.84
standard_deviation = standardDeviation(outcome_testingaccuracy)
print(standard_deviation)
print(sum(data) / len(data))
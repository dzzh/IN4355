import sys

TEST_FILE = 'test.txt'

attributes = []
trainingset = []
tests = []

status = 'undefined'

with open(TEST_FILE, 'r') as a_file:
	for a_line in a_file:
		if a_line.find('@attribute' != -1):
			status = 'attribute'
		elif a_line.find('@data' != -1):
			status = 'data'
		elif a_line.find('@test' != -1):
			status = 'test'
		else
			{
				'attribute': {
					
				}

			}[status]	

# training set, tests: list of dictionaries


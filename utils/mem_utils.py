from sys import getsizeof

def count(dictionary, c=0):
	for key in dictionary:
		if isinstance(dictionary[key], dict):
			# Calls repeatedly
			c = count(dictionary[key], c + 1)
		else:
			c += 1
	return c

def dict_size(dictionary, s=0):
	for key in dictionary:
		if isinstance(dictionary[key], dict):
			# Calls repeatedly
			s += dict_size(dictionary[key], getsizeof(dictionary[key]))
		else:
			s += getsizeof(dictionary[key])
	return s
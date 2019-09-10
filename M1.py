

print("This is name of M1 module: %s"%(__name__))


def printingStuff():
	print('There is  no stopping me without __name__=="__main__"')

if __name__ == '__main__':
	printingStuff()	
from specfile import PySpecFile
sf = PySpecFile("t.dat")
print(len(sf))
print(" Checking if deleted")
sf = None
print("passed")


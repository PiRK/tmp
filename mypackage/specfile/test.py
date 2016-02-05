from specfile import PySpecFile
sf = PySpecFile("t.dat")
print(len(sf))
print(sf.list())
print(sf.data(2))
print(" Checking if deleted")
sf = None
print("passed")

# testing memory usage:
#mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


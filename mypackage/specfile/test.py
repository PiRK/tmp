from specfile import PySpecFile
sf = PySpecFile("t.dat")
print("Status when opening t.dat: " + sf.get_error_string())
print("Number of scans: " + str(len(sf)))
print("List of scans\n: " + str(sf.list()))
sfdata = sf.data(2)
print("Data lines 2 to 5 in second scan:")
print(sfdata[2:5])
print("Shape of sfdata: " + str(sfdata.shape))

try:
    sf.data(108) # should raise IOError: Scan not found error (SpecFile)
except IOError as err:
    print(err)
    

sf = None  # testing destructor

sf2 = PySpecFile("doesnt_exist.dat")
print(dir(sf2))
print(sf2.get_error_string())


# testing memory usage:
#mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


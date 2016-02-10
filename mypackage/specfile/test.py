from specfile import SpecFile
sf = SpecFile("t.dat")
print("Status when opening t.dat: " + sf.get_error_string())
print("Number of scans: " + str(len(sf)))
print("List of scans\n: " + str(sf.list()))
sfdata = sf.data(2)
print("Data lines 2 to 5 in second scan:")
print(sfdata[2:5])
print("Shape of sfdata: " + str(sfdata.shape))
print("#N in scan 2: " + str(sf.columns(2)))
print("#S in scan 2: " + sf.command(2))

print("Scan index of scan 3.1: " + str(sf.index(3)))

try:
    sf.index(3, 2)  #scan 3.2 doesn't exist
except IOError as err:
    print("Trying to access scan 3.2")
    print(err)


try:
    sf.data(108) # should raise IOError: Scan not found error (SpecFile)
except IOError as err:
    print("Trying to access data of scan index 108")
    print(err)
    
sf = None  # testing destructor

sf2 = SpecFile("doesnt_exist.dat")
print("dummy")


# testing memory usage:
#mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


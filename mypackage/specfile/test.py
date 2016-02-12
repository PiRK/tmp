from specfile import SpecFile
sf = SpecFile("t.dat")
print("Status when opening t.dat: " + sf.get_error_string())
print("Number of scans: " + str(len(sf)))
print("List of scans\n: " + str(sf.list()))

scan2_1 = sf["2.1"]
scan2 = sf[2]

assert sf[2].header_dict['S'] == sf["2.1"].header_dict['S']
try:
   assert sf["3.1"].header_dict['S']  == sf[2].header_dict['S']
except AssertionError:
    print("""Assertion failed as expected""")
    
scan2 = sf[2]

print("\nData line 8 in second scan (from data line method):")
print(scan2.data_line(8))

print("\nData lines 2 to 5 in second scan (from data attribute):")
print(scan2.data[2:5])
print("Shape of scan2data: " + str(scan2.data.shape))
print("\n#N in scan 2: " + str(scan2.header_dict['N']))
print("#S in scan 2: " + scan2.header_dict['S'])
print("#L in scan 2: " + str(scan2.header_dict['L']))

print("\nList of all header lines in scan 2: ")
print(str(scan2.header_lines))
print("\nList of all file header lines related to scan 2: ")
print(str(scan2.file_header_lines))

print("\nScan index of scan 3.1: " + str(sf.index(3, 1)))
print("Scan number of 4th scan: " + str(sf.number(4)))
print("Scan order of 4th scan: " + str(sf.order(4)))

print("Trying to find index of scan 3.2")
try:
    sf.index(3, 2)  #scan 3.2 doesn't exist
except IOError as err:
    print(err)

print("\nTrying to access data of scan index 108 through SpecFile")
try:
    sf.data(108) # should raise IOError: Scan not found error (SpecFile)
except IOError as err:
    print(err)
    
print("\nTrying to parse scan 108")
try:
    scan108 = sf[108] # should raise IndexError: Scan index must be in range 1-100
except IndexError as err:
    print(err)
    
print("\nTrying to open a specfile that doesn't exist")
try:
    sf2 = SpecFile("doesnt_exist.dat")
except IOError as err:
    print(err)


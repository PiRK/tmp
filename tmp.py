import argparse
import pkgutil

import mypackage as m

parser = argparse.ArgumentParser(description='tmp interface script for mypackage')
parser.add_argument('command', help='command or module name')
parser.add_argument('mainargs', nargs='*', help='arguments passed to the module main function')


args = parser.parse_args()
print(args.command)
print(args.mainargs)

mods = [modname for _, modname, ispkg in pkgutil.iter_modules(m.__path__) if ispkg]

print(mods)
if args.command in mods:
    print("#todo from " + args.command + " import main; main" +
          str(tuple(args.mainargs)))
    
    
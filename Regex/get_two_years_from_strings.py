import re

def find_year(x):
    year1,year2 = re.findall(r'\b\w{4}\b', x)
    return year1,year2

# driver code 
x = input()
year1, year2 = find_year(x)
print(f'two year values in the given strings are {year1}   {year2}')
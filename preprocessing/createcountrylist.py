import csv

countries = {}

infile = open('owid-co2-data.csv', 'r')
outfile = open('countries.json', 'w')
for line in infile:
    country = line.split(',')[1]
    if not country in countries:
        countries.append(country)
        print(country)
infile.close()
outfile.close()
import csv
from collections import defaultdict
# Define the constant replacement value
cities = set()

id_to_location = defaultdict(str)
name_to_id = defaultdict(int)
with open('US Airline Flight Routes and Fares 1993-2024.csv', 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    # Get fieldnames from the reader to maintain header order
    fieldnames = reader.fieldnames
    for row in reader:
        name_to_id[row['city1']] = row['citymarketid_1']
        name_to_id[row['city2']] = row['citymarketid_2']
        cities.add((row['city1'], row['citymarketid_1']))
        cities.add((row['city2'], row['citymarketid_2']))
#         id1 = row['citymarketid_1']
#         id2 = row['citymarketid_2']

#         if row['citymarketid_1'] == "30194":
#             print(row['city1'], row['Geocoded_City1'])
#         if row['citymarketid_2'] == "30194":
#             print(row['city2'], row['Geocoded_City2'])
        
#         if row['Geocoded_City1']:
#             loc1 = row['Geocoded_City1'].split('\n')[-1]
#             id_to_location[id1].add(loc1)
#         if row['Geocoded_City2']:
#             loc2 = row['Geocoded_City2'].split('\n')[-1]
#             id_to_location[id2].add(loc2)
        

# for key, value in id_to_location.items():
#     print(f"ERROR: {key} -> {value}")

# number = len(cities)
# for name, idd in cities:
#     print(f"City {number}:\n{name}")
#     real = input("Enter Coordinate: ").strip()
#     id_to_location[idd] = {f"({real})"}
#     number -= 1
#     print("Saved as: ", id_to_location[idd])

# print()
with open("real_cities.txt", 'r') as file:
    while line:=file.readline():
        name = line.strip()
        location = file.readline().strip()
        idd = name_to_id[name]
        id_to_location[idd] = location
        print(f"Set {name} ({idd}) to {id_to_location[idd]}")

# Open the input CSV file and create a reader object
with open('USA_Filtered_Airline_2018-2024.csv', 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    # Get fieldnames from the reader to maintain header order
    fieldnames = reader.fieldnames


    # Open the output CSV file and create a writer object
    with open('USA_Filtered_Airline_2018-2024_FILLED.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through each row and apply conditions
        for row in reader:
            row['Geocoded_City1'] = id_to_location[row['citymarketid_1']]
            row['Geocoded_City2'] = id_to_location[row['citymarketid_2']]

            writer.writerow(row)

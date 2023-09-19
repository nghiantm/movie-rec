import csv

with open("movies.csv", "r+") as f:
    reader = csv.reader(f, delimiter=',')

    with open('output.csv', 'w', newline='') as output_file:
        reader = csv.reader(f)
        writer = csv.writer(output_file)

        for row in reader:
            if len(row) == 3:
                name = row[1]
                name = name.split()
                if len(name) > 1 and name[-2] == 'The' and name[1] and ',' in row[1]:
                    temp = "The"
                    name.remove("The")
                    new_s = " ".join(name)
                    newString = temp + ' ' + new_s
                    newString = newString.replace(',', '')
                    row[1] = newString
            else:
                pass
            writer.writerow(row)
    

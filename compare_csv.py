import csv

def read_and_sort_csv(file_path, skip_header):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        if skip_header:
            next(reader)  # Skip the header
        sorted_list = sorted(reader)
    return sorted_list

def compare_csv_files(file1, file2, skip_header=True):
    sorted_file1 = read_and_sort_csv(file1, skip_header)
    sorted_file2 = read_and_sort_csv(file2, skip_header)

    return sorted_file1 == sorted_file2

if __name__ == '__main__':
    file1 = 'data-1733497149273.csv'
    file2 = 'data-1733498530699.csv'

    if compare_csv_files(file1, file2):
        print("The CSV files are exactly the same.")
    else:
        print("The CSV files are different.")

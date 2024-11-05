import csv
import logging
from collections import defaultdict

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_counters():
    """Initialize counters for slices, samples, and classes."""
    return {
        "empty_slices": 0,
        "empty_samples": 0,
        "slice_count": 0,
        "sample_count": 0,
        "sample_has_classes": False
    }

def start_new_sample(counters, last_occurrence_per_sample, last_occurrences, last_occurrences_frequencies):
    """Update counters and record the last occurrence of the previous sample if needed."""
    counters["sample_count"] += 1
    if not counters["sample_has_classes"]:
        counters["empty_samples"] += 1
    counters["sample_has_classes"] = False
    if last_occurrence_per_sample is not None:
        record_occurrences(last_occurrences, last_occurrences_frequencies, last_occurrence_per_sample)

def process_slice(row, slice_data):
    """Append a row to the current slice and check if a complete slice has been formed."""
    slice_data.append(row)
    if len(slice_data) == 3:
        return slice_data
    return None

def check_empty_slice(slice_data):
    """Determine if a slice is empty by checking class labels."""
    return slice_data[0][2] == "" and slice_data[1][2] == "" and slice_data[2][2] == ""

def update_occurrences(slice_index, first_occurrence_per_sample, last_occurrence_per_sample):
    """Set or update the first and last occurrence indices for a sample."""
    if first_occurrence_per_sample is None:
        first_occurrence_per_sample = slice_index
    last_occurrence_per_sample = slice_index
    return first_occurrence_per_sample, last_occurrence_per_sample

def record_occurrences(occurrences_list, occurrences_freq, occurrence):
    """Append an occurrence to a list and update its frequency map."""
    occurrences_list.append(occurrence)
    occurrences_freq[occurrence] += 1

def save_frequency_map(filename, frequency_map):
    """Save a frequency map to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Frequency'])
        for index, frequency in sorted(frequency_map.items()):
            writer.writerow([index, frequency])

def save_vector(filename, vector):
    """Save a plain list as a single-row CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(vector)

def process_csv_file(csv_file_path):
    """Read and process the CSV file row by row, tracking occurrences and empty slices."""
    counters = initialize_counters()
    first_occurrences = []
    last_occurrences = []
    first_occurrences_frequencies = defaultdict(int)
    last_occurrences_frequencies = defaultdict(int)

    first_occurrence_per_sample = None
    last_occurrence_per_sample = None
    slice_data = []

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Discard CSV header

        for row in reader:
            slice_index = int(row[0].split("_")[3])

            # Check for the start of a new sample
            if slice_index == 1 and not slice_data:
                start_new_sample(counters, last_occurrence_per_sample, last_occurrences, last_occurrences_frequencies)
                first_occurrence_per_sample = None
                last_occurrence_per_sample = None

            processed_slice = process_slice(row, slice_data)
            if not processed_slice:
                continue

            # Update counters and process occurrences
            counters["slice_count"] += 1
            if check_empty_slice(processed_slice):
                counters["empty_slices"] += 1
            else:
                counters["sample_has_classes"] = True
                first_occurrence_per_sample, last_occurrence_per_sample = update_occurrences(
                    slice_index, first_occurrence_per_sample, last_occurrence_per_sample
                )

                # Record the first occurrence for the sample
                if first_occurrence_per_sample == slice_index:
                    record_occurrences(first_occurrences, first_occurrences_frequencies, slice_index)

            slice_data.clear()

        # Record the last occurrence for the final sample
        if last_occurrence_per_sample is not None:
            record_occurrences(last_occurrences, last_occurrences_frequencies, last_occurrence_per_sample)

    logging.info(f"{counters['empty_slices']}/{counters['slice_count']} empty slices and {counters['empty_samples']}/{counters['sample_count']} empty samples")
    return first_occurrences, last_occurrences, first_occurrences_frequencies, last_occurrences_frequencies

def undersample_3d(csv_file_path='./data/train.csv'):
    """Main function to process the 3D data, track occurrences, and save results to CSV files."""
    first_occurrences, last_occurrences, first_occurrences_frequencies, last_occurrences_frequencies = process_csv_file(csv_file_path)

    # Save all data to CSV files
    save_frequency_map('first_occurrences_frequency.csv', first_occurrences_frequencies)
    save_frequency_map('last_occurrences_frequency.csv', last_occurrences_frequencies)
    save_vector('first_occurrences_vector.csv', first_occurrences)
    save_vector('last_occurrences_vector.csv', last_occurrences)

# Main application code
if __name__ == "__main__":
    undersample_3d()

import cdflib
import sys

# This script will check the contents of a single CDF file.
# The file path has been corrected to point to the right location inside the L1_TH2 folder.

file_to_check = r"C:\Users\Abhijith P\Desktop\HALO CME PROJECT\raw_data\L1_TH2\AL1_ASW91_L2_TH2_20240910_UNP_9999_999999_V02.cdf"

print(f"Attempting to read as a CDF file: {file_to_check}")

try:
    # Open the specified file using cdflib
    with cdflib.CDF(file_to_check) as cdf:
        print(f"\n--- Variables in: {file_to_check} ---")
        
        # Print the metadata
        print(cdf.cdf_info())
        
        print("\n--- End of variables ---")

except Exception as e:
    print(f"\nAn error occurred while trying to read the file: {e}")
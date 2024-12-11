# Generate missing provenance from given log files

This repository is a side tool to generate missing provenance from given log/csv files. 
The tool main tool is `generate_provenance.py` which reads the log files and generates the missing provenance in sql. 
The output file can be run directly in the database to update the provenance table and resources tables. 

## Project Structure
- `generate_provenance.py`: The main tool to generate missing provenance from log files.
- `log_files/`: Directory containing the log files.
- `output/`: Directory containing the generated sql files.
- All given input files are put in the same directory as the script.
- `trail_path.py`: Helper script to show a full path of a given trail, either from a given `prov_id` or HaNA number. 

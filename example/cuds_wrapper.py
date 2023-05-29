import os
import re


def timestep(input, output):

    with open(input, 'r') as fin:
        for line in fin:
            if 'timestep' in line:
                _, value = line.split('=')
                break

    string = ['variable ts      equal' + value]
    variable_string = '# Input variables.\n'
    with open(output, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()


def temperature(input, output):

    with open(input, 'r') as fin:
        for line in fin:
            if 'temperature' in line:
                _, value = line.split('=')
                break

    string = ['variable tf     equal' + value]
    variable_string = '# Input variables.\n'
    with open(output, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()


def pressure(input, output):

    with open(input, 'r') as fin:
        for line in fin:
            if 'pressure' in line:
                _, value = line.split('=')
                break
    string = ['variable p      equal' + value]
    variable_string = '# Input variables.\n'
    with open(output, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()

def correlation_length(input, output):

    with open(input, 'r') as fin:
        for line in fin:
            if 'correlation_length' in line:
                _, value = line.split('=')
                break
    string = ['variable cl     equal' + value]
    variable_string = '# Input variables.\n'
    with open(output, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()

def sample_interval(input, output):

    with open(input, 'r') as fin:
        for line in fin:
            if 'sample_interval' in line:
                _, value = line.split('=')
                break
    string = ['variable s     equal' + value]
    variable_string = '# Input variables.\n'
    with open(output, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()

def production_step(input, output):

    with open(input, 'r') as fin:
        for line in fin:
            if 'production_step' in line:
                _, value = line.split('=')
                break
    string = ['variable prod     equal' + value]
    variable_string = '# Input variables.\n'
    with open(output, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()


# correlation_length('setting.db', 'water_aa.lt')

def update_input_file(variable_name, input_file, output_file):
    with open(input_file, 'r') as fin:
        for line in fin:
            if variable_name in line:
                _, value = line.split('=')
                break

    string = ['variable ' + variable_name + '     equal' + value]
    variable_string = '# Input variables.\n'

    with open(output_file, 'r+') as fout:
        for t in string:
            replace = fout.read().replace(variable_string, variable_string + t)
            fout.seek(0)
            fout.write(replace)
            fout.close()

def process_log_file(log_file):
    # Read log file
    with open(log_file, 'r') as file:
        log_contents = file.read()

    # Extract T value using regular expression
    T_match = re.search(r'\s*fix\s*1\s*all\s*nvt.*?(\S+)', log_contents)
    T = T_match.group(1) if T_match else None

    # Extract n_run value using regular expression
    n_run_match = re.search(r'run\s+(\S+)', log_contents)
    n_run = n_run_match.group(1) if n_run_match else None

    # Extract thermo data using regular expression and write to thermo.dat file
    thermo_data_match = re.search(r'Step.*?Loop time.*?', log_contents, re.DOTALL)
    thermo_data = thermo_data_match.group(0) if thermo_data_match else ''
    with open('thermo.dat', 'w') as file:
        file.write(thermo_data)

    return T, n_run




def get_data(input, property, start, stop):
    import pandas as pd
    data = pd.read_csv(input, sep='\s+', nrows=start, skiprows=stop)
    df = pd.DataFrame(data)
    return df[property]


def extract_variable_value(file_path, variable_name):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('variable'):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == variable_name:
                    return parts[3]
    return None




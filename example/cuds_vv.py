# from cuds_wrapper import get_data, process_log_file
from cuds_wrapper import get_data, process_log_file, extract_variable_value
from simphony_osp.tools import export_file, import_file, pretty_print, search
from simphony_osp.tools.search import sparql
from simphony_osp.tools.pico import install, namespaces, packages, uninstall
from simphony_osp.session import Session, core_session
from simphony_osp.namespaces import miso
from simphony_osp.tools import semantic2dot


# TODO: 1) Pretty hard coded, need to soft code this
#       2) delta method is a dummy example at the moment, need to refine
#       3) also V&V part should be in a sepreate script/ function and called here
install('miso.yml')

#Strip lammps.log for simualtion data
process_log_file('lammps.log')

# Extract relevant data from input 
density_data = get_data('thermo.dat', 'Density',10, 0)
step_data = get_data('thermo.dat', 'Step', 10, 0)

# Extract simulation input variables from moltemplate file

moltemplate_file = 'water_aa.lt'
tf = extract_variable_value(moltemplate_file, 'tf')
p = extract_variable_value(moltemplate_file, 'p')
ts = extract_variable_value(moltemplate_file, 'ts')
cl =  extract_variable_value(moltemplate_file, 'cl')


#Create a session
session = Session()

# Create a simulation object
simulation = miso.Simulation()

#Create an object for inputs
sim_input = miso.Input()

#Define material
material = miso.Material()

#Define the system
water = miso.Water()
O = miso.O()
H = miso.H()
water[miso.hasPart] = {O,H}
material[miso.hasPart] = water

#Make a model

model = miso.Model()

#Define variables and parameters
input_temperature = miso.Temperature()
input_pressure = miso.Pressure()
input_time_step = miso.Time()
input_correlation_length = miso.CorrelationLength()



#Assign input value
temperature_value = miso.Value(quantity=tf)
pressure_value = miso.Value(quantity=p)
time_step_value = miso.Value(quantity=ts)
correlation_length_value = miso.Value(quantity=cl)

input_temperature[miso.hasValue] = temperature_value
input_pressure[miso.hasValue] = pressure_value
input_time_step[miso.hasValue] = time_step_value
input_correlation_length[miso.hasValue] = correlation_length_value

model[miso.hasPart] = {input_temperature, input_pressure, input_time_step, input_correlation_length}

#Define computational method
comp_method = miso.MolecularDynamics()
model[miso.hasPart] = comp_method

#Define materials relation
mat_rel = miso.OPLS()
model[miso.hasPart] = mat_rel

#Define physics equation
phys_eq = miso.NewtonEquation()
model[miso.hasPart] = phys_eq

# #Define engine
# lammps = miso.LAMMPS()
# model[miso.hasPart] = lammps

#Add model and material to input
sim_input[miso.hasPart] = material
sim_input[miso.hasModel] = model



#Define outputs

sim_output = miso.Output()

#Add density information
density = miso.Density()
value = miso.Value(quantity=density_data)
unit = miso.CustomUnit(string='g/cm^3')
density[miso.hasValue] = value
density[miso.hasUnit] = unit

#Add simulation step information
step = miso.Step()
step_value = miso.Value(quantity=step_data)
step[miso.hasValue] = step_value
density[miso.hasPart] = step

sim_output[miso.hasPart] = density

#Add to simulation
simulation[miso.hasInput] = sim_input
simulation[miso.hasOutput] = sim_output

session.add(simulation)

#Commit cuds to session


session.commit()



# Query density data using SPARQL

result = sparql(
    f"""
    SELECT ?densityValue WHERE {{
        ?density rdf:type <{miso.Density.identifier}> .
        ?density <{miso.hasValue}> ?value .
        ?value <{miso.quantity}> ?densityValue .
    }}
    """
)

#Store the queried simulation density values and use for delta method

density_values = []

for row in result:
    density_value_str = row['densityValue']
    lines = density_value_str.split('\n')
    for line in lines:
        values = line.strip().split()
        if len(values) > 1 and values[0].isdigit():
            density_values.append(float(values[1]))

print(density_values)


# Plot simulation density from lamps against experimental values

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

x = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
y1 = [0.69, 0.9656, 0.9656, 0.89, 0.932,0.911,0.9729, 0.975, 0.92326574, 0.852]
# y1 = [0.89] * len(x)


plt.scatter(x, density_values, label='Simulation')
plt.scatter(x, y1, label='Experiment')
plt.legend()
plt.show()

# calculate differences for delta method
diffs = []

diffs = np.array(y1) - np.array(density_values)


print(diffs)
x_train = np.arange(len(diffs)).reshape(-1, 1)
y_train = diffs.reshape(-1, 1)

# Using Bayesian regression for training

model = BayesianRidge(tol=1e-3, fit_intercept=True, compute_score=True, n_iter=500)
model.fit(x_train, y_train)

# Predict on the training set
y_pred = model.predict(x_train)

# Plot the training set and the prediction
plt.scatter(x_train, y_train, label='Training set')
plt.scatter(x_train, y_pred, color='red', label='Prediction')
plt.xlabel('Index')
plt.ylabel('Difference')
plt.legend()
plt.show()



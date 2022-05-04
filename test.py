# import packages
from pystoned import wCNLS, DEA, CNLS
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS, ORIENT_OO
import numpy as np
import pandas as pd


# set seed
np.random.seed(0)

# generate DMUs: DGP
#x = np.random.uniform(low=1, high=10, size=(100, 2))
x1 = np.random.uniform(low=1, high=10, size=(500))
x2 = np.random.uniform(low=1, high=10, size=(500))
u = np.random.normal(loc=0, scale=0.7, size=500)
y = x1**0.4*x2**0.5+u

x = np.concatenate((x1.reshape(len(x1), 1), x2.reshape(len(x2), 1)), axis=1)

grid_size_x1 = 10
grid_size_x2 = 10
grid_size_y = 10

data = pd.DataFrame()

bagged_data = pd.DataFrame()

# Getting the max density and flow values to calculcate the size of the bag

datax1 = pd.DataFrame(x1, columns=['x1'])
datax2 = pd.DataFrame(x2, columns=['x2'])
datay = pd.DataFrame(y, columns=['y'])

data = pd.concat([datax1, datax2, datay], axis=1)

# Getting the max density and flow values to calculcate the size of the bag
max_x1 = data.x1.max()
max_x2 = data.x2.max()
max_y = data.y.max()

# Calclulating the size of the bag
grid_x1_scale = max_x1 / grid_size_x1
grid_x2_scale = max_x2 / grid_size_x2
grid_y_scale = max_y / grid_size_y

# Assigning the bag number for density and
data['grid_x1'] = data.x1 / grid_x1_scale
data['grid_x2'] = data.x2 / grid_x2_scale
data['grid_y'] = data.y / grid_y_scale

data = data.astype({'grid_x1': int, 'grid_x2': int, 'grid_y': int})

# Calculating the centroid and the weight of each bag
bagged_data = data.groupby(
    ['grid_x1', 'grid_x2', 'grid_y'],
    as_index=False).agg(
        bag_size=('grid_x1', 'count'), sum_x1=('x1', 'sum'), sum_x2=('x2', 'sum'), sum_y=('y', 'sum'))
bagged_data['centroid_x1'] = bagged_data.sum_x1.div(bagged_data.bag_size)
bagged_data['centroid_x2'] = bagged_data.sum_x2.div(bagged_data.bag_size)
bagged_data['centroid_y'] = bagged_data.sum_y.div(bagged_data.bag_size)
bagged_data['weight'] = bagged_data.bag_size.div(len(data))


nx1 = np.array(bagged_data.centroid_x1)
nx2 = np.array(bagged_data.centroid_x2)
ny = np.array(bagged_data.centroid_y)
nw = np.array(bagged_data.weight)

nx = np.concatenate((nx1.reshape(len(nx1), 1), nx2.reshape(len(nx2), 1)), axis=1)

model1 = wCNLS.wCNLS(y=ny, x=nx, w=nw, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
model1.optimize()
yhat1 = model1.get_frontier()
mse1 = (np.array(yhat1)-ny)**2

model2 = CNLS.CNLS(y=y, x=x, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
model2.optimize()
yhat2 = model2.get_frontier()
mse2 = (np.array(yhat2)-y)**2

np.savetxt("mse1.csv", mse1, delimiter=",")
np.savetxt("mse2.csv", mse2, delimiter=",")

"""
# define and solve the DEA radial model
model2 = DEA.DEA(y=yhat1, x=nx, rts=RTS_VRS, orient=ORIENT_OO, yref=None, xref=None)
model2.optimize(OPT_LOCAL)

# display the technical efficiency
model2.display_theta()
"""